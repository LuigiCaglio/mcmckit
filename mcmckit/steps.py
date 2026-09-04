"""Single-step sampling functions - you own the loop.

Every function here advances a chain by exactly one step. State goes in as
plain arguments and comes back as plain return values; nothing is stored on
an object, so you write the recursion yourself::

    x = x0
    logp = log_post(x)
    S = np.linalg.cholesky(np.eye(d) * 0.1**2)

    for i in range(1, n_iter + 1):
        x, logp, S, accepted = mc.ram_step(log_post, x, logp, S, i)
        chain[:, i] = x
        # ... your convergence check, your logging, your early stop ...

The full-run helpers in :mod:`mcmckit.runners` are thin loops over these
functions, so both interfaces run identical code.

Sign convention
---------------
Every function takes a **log-posterior**: larger means a better fit. If your
code carries a negative log-posterior (an ``nll``), wrap it once::

    log_post = lambda theta: -my_nll(theta)

Auxiliary model output
----------------------
If your callable returns a tuple ``(log_post_value, aux)``, the extra payload
is threaded back out untouched as the last return value, and you pass the
current one back in via ``aux=``. Use it to keep whatever your forward model
already computed - natural frequencies, mode shapes, residuals - without
re-running it::

    def log_post(theta):
        freqs = surrogate(theta)
        return -0.5 * np.sum(((freqs - measured) / sigma) ** 2), freqs

    x, logp, S, accepted, freqs = mc.ram_step(log_post, x, logp, S, i, aux=freqs)

If the callable returns a plain float, no aux is returned and the signature
stays its usual width.
"""

from typing import NamedTuple

import numpy as np

__all__ = [
    "mh_step",
    "mala_step",
    "ram_step",
    "adaptive_mala_step",
    "dram_step",
    "gibbs_step",
    "DRAMState",
    "init_dram_state",
    "as_cov",
]


# ======================================================================
# Helpers
# ======================================================================

def as_cov(cov, d):
    """Broadcast a scalar, 1-D diagonal or full matrix to a (d, d) covariance.

    Parameters
    ----------
    cov : float or array-like
        Scalar (isotropic variance), 1-D (diagonal), or 2-D (full matrix).
    d : int
        Dimension of the parameter space.

    Returns
    -------
    ndarray, shape (d, d)
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = np.eye(d) * float(cov)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    if cov.shape != (d, d):
        raise ValueError(f"covariance shape {cov.shape} incompatible with dimension {d}")
    return cov


def _call(log_post, x):
    """Evaluate log_post, splitting an optional auxiliary payload.

    Returns ``(value, aux, has_aux)``. ``has_aux`` is True only when the
    callable returned a tuple, which is the documented opt-in signal.
    """
    out = log_post(x)
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError(
                f"log-posterior returned a {len(out)}-tuple; expected either a "
                "float or a 2-tuple (log_post_value, aux)."
            )
        return float(out[0]), out[1], True
    return float(out), None, False


def _pack(has_aux, aux, *values):
    """Append aux to the return tuple only when the callable opted in."""
    return (*values, aux) if has_aux else values


def _rng(rng):
    return np.random.default_rng() if rng is None else rng


# ======================================================================
# Metropolis-Hastings
# ======================================================================

def mh_step(log_post, x, logp, cov, aux=None, rng=None):
    """One random-walk Metropolis-Hastings step.

    Parameters
    ----------
    log_post : callable
        ``theta -> log posterior`` (unnormalised), or ``theta -> (log
        posterior, aux)``.
    x : ndarray, shape (d,)
        Current position.
    logp : float
        ``log_post(x)``, threaded in so it is never recomputed.
    cov : array-like
        Proposal covariance. Scalar, 1-D diagonal or full ``(d, d)``.
    aux : object, optional
        Auxiliary payload belonging to ``x``, threaded through on rejection.
    rng : numpy.random.Generator, optional

    Returns
    -------
    x_new, logp_new, accepted
        Plus ``aux_new`` as a fourth value if ``log_post`` returns a tuple.
    """
    rng = _rng(rng)
    x = np.asarray(x, dtype=float)
    d = x.shape[0]
    L = np.linalg.cholesky(as_cov(cov, d))

    proposal = x + L @ rng.standard_normal(d)
    logp_prop, aux_prop, has_aux = _call(log_post, proposal)

    if np.log(rng.random()) < logp_prop - logp:
        return _pack(has_aux, aux_prop, proposal, logp_prop, True)
    return _pack(has_aux, aux, x, logp, False)


# ======================================================================
# Robust Adaptive Metropolis (Vihola 2012)
# ======================================================================

def ram_step(log_post, x, logp, S, i, gamma=0.51, target_rate=0.234,
             aux=None, rng=None):
    """One Robust Adaptive Metropolis step (Vihola 2012).

    The adaptation state is the Cholesky factor ``S``, threaded in and out
    explicitly. Seed it with ``np.linalg.cholesky(as_cov(cov0, d))``.

    Parameters
    ----------
    log_post : callable
        ``theta -> log posterior``, or ``theta -> (log posterior, aux)``.
    x : ndarray, shape (d,)
        Current position.
    logp : float
        ``log_post(x)``.
    S : ndarray, shape (d, d)
        Lower-triangular Cholesky factor of the proposal covariance.
    i : int
        Step number, 1-indexed. Drives the adaptation decay ``i ** -gamma``.
    gamma : float
        Adaptation decay exponent, in (0.5, 1]. Default 0.51.
    target_rate : float
        Target acceptance rate. Default 0.234.
    aux : object, optional
        Auxiliary payload belonging to ``x``.
    rng : numpy.random.Generator, optional

    Returns
    -------
    x_new, logp_new, S_new, accepted
        Plus ``aux_new`` as a fifth value if ``log_post`` returns a tuple.
    """
    if i < 1:
        raise ValueError(f"step index i must be 1-indexed and >= 1, got {i}")
    rng = _rng(rng)
    x = np.asarray(x, dtype=float)
    S = np.asarray(S, dtype=float)
    d = x.shape[0]

    r = rng.standard_normal(d)
    proposal = x + S @ r
    logp_prop, aux_prop, has_aux = _call(log_post, proposal)

    alpha = min(1.0, np.exp(min(logp_prop - logp, 0.0)))

    # Rank-1 Cholesky adaptation of the proposal scale.
    r_sq = r @ r
    if r_sq > 0:
        eta = i ** (-gamma)
        rank1 = np.eye(d) + eta * (alpha - target_rate) * np.outer(r, r) / r_sq
        try:
            S = np.linalg.cholesky(S @ rank1 @ S.T)
        except np.linalg.LinAlgError:
            pass  # keep the previous S if the update loses positive definiteness

    if rng.random() < alpha:
        return _pack(has_aux, aux_prop, proposal, logp_prop, S, True)
    return _pack(has_aux, aux, x, logp, S, False)


# ======================================================================
# MALA
# ======================================================================

def _mala_log_q(x_from, x_to, grad_from, eps):
    """Log density of proposing x_to from x_from under a Langevin kernel."""
    mean = x_from + 0.5 * eps**2 * grad_from
    diff = x_to - mean
    return -0.5 / eps**2 * (diff @ diff)


def mala_step(log_post_and_grad, x, logp, grad, step_size, aux=None, rng=None):
    """One Metropolis-adjusted Langevin step.

    Parameters
    ----------
    log_post_and_grad : callable
        ``theta -> (log posterior, gradient)``, or
        ``theta -> (log posterior, gradient, aux)``.
    x : ndarray, shape (d,)
        Current position.
    logp : float
        Log posterior at ``x``.
    grad : ndarray, shape (d,)
        Gradient of the log posterior at ``x``, threaded so it is not
        recomputed.
    step_size : float
        Langevin step size. Tune for roughly 0.57 acceptance.
    aux : object, optional
    rng : numpy.random.Generator, optional

    Returns
    -------
    x_new, logp_new, grad_new, accepted
        Plus ``aux_new`` as a fifth value if the callable returns a 3-tuple.
    """
    rng = _rng(rng)
    x = np.asarray(x, dtype=float)
    grad = np.asarray(grad, dtype=float)
    d = x.shape[0]
    eps = float(step_size)

    proposal = x + 0.5 * eps**2 * grad + eps * rng.standard_normal(d)

    out = log_post_and_grad(proposal)
    if len(out) == 3:
        logp_prop, grad_prop, aux_prop = out
        has_aux = True
    else:
        (logp_prop, grad_prop), aux_prop, has_aux = out, None, False
    grad_prop = np.asarray(grad_prop, dtype=float)

    log_alpha = (logp_prop + _mala_log_q(proposal, x, grad_prop, eps)
                 - logp - _mala_log_q(x, proposal, grad, eps))

    if np.log(rng.random()) < log_alpha:
        return _pack(has_aux, aux_prop, proposal, logp_prop, grad_prop, True)
    return _pack(has_aux, aux, x, logp, grad, False)


def adaptive_mala_step(log_post_and_grad, x, logp, grad, log_step, i,
                       gamma=0.51, target_rate=0.574, aux=None, rng=None):
    """One MALA step with log-space step-size adaptation.

    Identical to :func:`mala_step` except the step size adapts itself. The
    adaptation state is ``log_step``, threaded in and out. Seed it with
    ``np.log(initial_step_size)``.

    Parameters
    ----------
    log_step : float
        Log of the current Langevin step size.
    i : int
        Step number, 1-indexed.
    gamma : float
        Adaptation decay exponent. Default 0.51.
    target_rate : float
        Target acceptance rate. Default 0.574, optimal for MALA.

    Returns
    -------
    x_new, logp_new, grad_new, log_step_new, accepted
        Plus ``aux_new`` as a sixth value if the callable returns a 3-tuple.
    """
    if i < 1:
        raise ValueError(f"step index i must be 1-indexed and >= 1, got {i}")
    rng = _rng(rng)
    x = np.asarray(x, dtype=float)
    grad = np.asarray(grad, dtype=float)
    d = x.shape[0]
    eps = np.exp(log_step)

    proposal = x + 0.5 * eps**2 * grad + eps * rng.standard_normal(d)

    out = log_post_and_grad(proposal)
    if len(out) == 3:
        logp_prop, grad_prop, aux_prop = out
        has_aux = True
    else:
        (logp_prop, grad_prop), aux_prop, has_aux = out, None, False
    grad_prop = np.asarray(grad_prop, dtype=float)

    log_alpha = (logp_prop + _mala_log_q(proposal, x, grad_prop, eps)
                 - logp - _mala_log_q(x, proposal, grad, eps))
    alpha = min(1.0, np.exp(min(log_alpha, 0.0)))

    log_step = log_step + i ** (-gamma) * (alpha - target_rate)

    if rng.random() < alpha:
        return _pack(has_aux, aux_prop, proposal, logp_prop, grad_prop, log_step, True)
    return _pack(has_aux, aux, x, logp, grad, log_step, False)


# ======================================================================
# DRAM
# ======================================================================

class DRAMState(NamedTuple):
    """Adaptation state for :func:`dram_step`.

    Plain data, no behaviour. ``C`` is the live proposal covariance; ``mean``
    and ``M2`` accumulate the empirical covariance recursively (Welford), so a
    step costs O(d^2) regardless of how long the chain gets.

    Attributes
    ----------
    C : ndarray, shape (d, d)
        Current proposal covariance.
    mean : ndarray, shape (d,)
        Running mean of the visited states.
    M2 : ndarray, shape (d, d)
        Running sum of squared deviations. Empirical covariance is M2 / (n-1).
    n : int
        Number of states folded into mean and M2.
    """

    C: np.ndarray
    mean: np.ndarray
    M2: np.ndarray
    n: int


def init_dram_state(x0, initial_cov=None):
    """Build the starting :class:`DRAMState` for a chain at ``x0``."""
    x0 = np.asarray(x0, dtype=float)
    d = x0.shape[0]
    C = as_cov(0.1**2 if initial_cov is None else initial_cov, d)
    return DRAMState(C=C, mean=np.zeros(d), M2=np.zeros((d, d)), n=0)


def dram_step(log_post, x, logp, state, dr_scale=0.1, adapt_start=100,
              adapt_interval=10, regularization=1e-6, aux=None, rng=None):
    """One Delayed Rejection Adaptive Metropolis step.

    Two-stage delayed rejection with Haario adaptive-Metropolis covariance
    learning. The adaptation state is a :class:`DRAMState`, threaded in and
    out; build the first one with :func:`init_dram_state`.

    Parameters
    ----------
    log_post : callable
        ``theta -> log posterior``, or ``theta -> (log posterior, aux)``.
    x : ndarray, shape (d,)
        Current position.
    logp : float
        ``log_post(x)``.
    state : DRAMState
        Adaptation state carried between steps.
    dr_scale : float
        The second-stage proposal uses ``dr_scale**2 * C``. Default 0.1.
    adapt_start : int
        Steps to take before the covariance starts adapting. Default 100.
    adapt_interval : int
        Adapt every this many steps. Default 10.
    regularization : float
        Diagonal jitter keeping the adapted covariance positive definite.
    aux : object, optional
    rng : numpy.random.Generator, optional

    Returns
    -------
    x_new, logp_new, state_new, accepted
        Plus ``aux_new`` as a fifth value if ``log_post`` returns a tuple.
        ``accepted`` is 0 (rejected), 1 (accepted at stage 1) or 2 (accepted
        at stage 2), so it stays truthy on acceptance while telling you which
        stage did the work.
    """
    rng = _rng(rng)
    x = np.asarray(x, dtype=float)
    d = x.shape[0]
    C = state.C

    # ---- stage 1 -----------------------------------------------------
    L1 = np.linalg.cholesky(C)
    theta1 = x + L1 @ rng.standard_normal(d)
    logp1, aux1, has_aux = _call(log_post, theta1)

    alpha1 = min(1.0, np.exp(min(logp1 - logp, 0.0)))

    accepted = 0
    if rng.random() < alpha1:
        x_new, logp_new, aux_new, accepted = theta1, logp1, aux1, 1
    else:
        # ---- stage 2: a smaller, more conservative retry --------------
        L2 = np.linalg.cholesky(C * dr_scale**2)
        theta2 = x + L2 @ rng.standard_normal(d)
        logp2, aux2, _ = _call(log_post, theta2)

        alpha1_prime = min(1.0, np.exp(min(logp1 - logp2, 0.0)))

        # log q1(theta1|theta2) - log q1(theta1|x), symmetric Gaussian q1(., C)
        diff2 = np.linalg.solve(L1, theta1 - theta2)
        diff_curr = np.linalg.solve(L1, theta1 - x)
        log_prop_ratio = -0.5 * (diff2 @ diff2 - diff_curr @ diff_curr)

        log_alpha2 = (logp2 - logp + log_prop_ratio
                      + np.log(max(1.0 - alpha1_prime, 1e-300))
                      - np.log(max(1.0 - alpha1, 1e-300)))

        if np.log(rng.random()) < log_alpha2:
            x_new, logp_new, aux_new, accepted = theta2, logp2, aux2, 2
        else:
            x_new, logp_new, aux_new = x, logp, aux

    # ---- recursive mean / covariance update (Welford) -----------------
    n = state.n + 1
    delta = x_new - state.mean
    mean = state.mean + delta / n
    M2 = state.M2 + np.outer(delta, x_new - mean)

    # ---- adapt the proposal covariance -------------------------------
    if n >= adapt_start and n % adapt_interval == 0 and n > 1:
        emp_cov = M2 / (n - 1)
        C = (2.38**2 / d) * emp_cov + regularization * np.eye(d)

    new_state = DRAMState(C=C, mean=mean, M2=M2, n=n)
    return _pack(has_aux, aux_new, x_new, logp_new, new_state, accepted)


# ======================================================================
# Metropolis-within-Gibbs
# ======================================================================

def gibbs_step(log_post, x, logp, blocks, proposal_std, aux=None, rng=None):
    """One Metropolis-within-Gibbs sweep, updating every block once.

    Parameters
    ----------
    log_post : callable
        ``theta -> log posterior``, or ``theta -> (log posterior, aux)``.
    x : ndarray, shape (d,)
        Current position.
    logp : float
        ``log_post(x)``.
    blocks : sequence of sequence of int
        Index groups updated together. ``[[0], [1], [2]]`` is scalar Gibbs;
        ``[[0, 1], [2]]`` updates the first two jointly.
    proposal_std : float or sequence of float
        Random-walk width, shared or one per block.
    aux : object, optional
    rng : numpy.random.Generator, optional

    Returns
    -------
    x_new, logp_new, accepted
        ``accepted`` is a list with one 0/1 flag per block, in block order.
        Plus ``aux_new`` as a fourth value if ``log_post`` returns a tuple.
    """
    rng = _rng(rng)
    x = np.asarray(x, dtype=float).copy()

    blocks = [list(b) for b in blocks]
    if np.isscalar(proposal_std):
        stds = [float(proposal_std)] * len(blocks)
    else:
        stds = list(proposal_std)
        if len(stds) != len(blocks):
            raise ValueError(
                f"proposal_std has {len(stds)} entries but there are "
                f"{len(blocks)} blocks."
            )

    has_aux = False
    accepted = []
    for block, std in zip(blocks, stds):
        proposal = x.copy()
        proposal[block] += rng.standard_normal(len(block)) * std

        logp_prop, aux_prop, has_aux = _call(log_post, proposal)

        if np.log(rng.random()) < logp_prop - logp:
            x, logp, aux = proposal, logp_prop, aux_prop
            accepted.append(1)
        else:
            accepted.append(0)

    return _pack(has_aux, aux, x, logp, accepted)
