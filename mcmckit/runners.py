"""Full-run helpers - thin loops over :mod:`mcmckit.steps`.

Use these when you just want the chain and do not need to touch the
recursion. Every one of them is a short loop over the matching step function,
so the two interfaces run identical code::

    result = mc.ram(log_post, x0=[0.0, 0.0], n_samples=10_000)
    print(result.mean())
    result.discard(1000).plot_corner()

If you want control of the loop - an early stop, a custom convergence check,
saving to disk as you go, a progress bar of your own - call the step
functions directly instead. See :mod:`mcmckit.steps`.
"""

import numpy as np

from .core.result import Result
from .steps import (
    DRAMState,
    adaptive_mala_step,
    as_cov,
    dram_step,
    gibbs_step,
    init_dram_state,
    mala_step,
    mh_step,
    ram_step,
)

__all__ = ["metropolis", "ram", "mala", "adaptive_mala", "dram", "gibbs"]


def _resolve(log_post, param_names):
    """Accept either a bare callable or a Problem, and pick up param names."""
    if hasattr(log_post, "log_posterior"):
        problem = log_post
        if param_names is None:
            param_names = getattr(problem, "param_names", None)
        return problem.log_posterior, param_names
    return log_post, param_names


def _resolve_grad(log_post, param_names):
    """Same, for samplers that need the gradient alongside the density."""
    if hasattr(log_post, "log_posterior_and_grad"):
        problem = log_post
        if param_names is None:
            param_names = getattr(problem, "param_names", None)
        if not getattr(problem, "has_grad", False):
            raise RuntimeError(
                "This sampler needs gradients: pass grad_log_likelihood and "
                "grad_log_prior to Problem, or hand in a callable returning "
                "(log_post, grad)."
            )
        return problem.log_posterior_and_grad, param_names
    return log_post, param_names


def _finish(chain, logps, param_names, acceptance_rate):
    return Result(
        samples=np.asarray(chain),
        log_posteriors=np.asarray(logps),
        param_names=param_names,
        acceptance_rate=acceptance_rate,
    )


def metropolis(log_post, x0, n_samples, proposal_cov, param_names=None, rng=None):
    """Run random-walk Metropolis-Hastings to completion.

    Parameters
    ----------
    log_post : callable or Problem
        ``theta -> log posterior``, or a :class:`~mcmckit.Problem`.
    x0 : array-like, shape (d,)
        Starting point.
    n_samples : int
        Number of steps to take.
    proposal_cov : float or array-like
        Fixed proposal covariance.
    param_names : sequence of str, optional
    rng : numpy.random.Generator, optional

    Returns
    -------
    Result
    """
    log_post, param_names = _resolve(log_post, param_names)
    x = np.asarray(x0, dtype=float)
    cov = as_cov(proposal_cov, x.shape[0])
    logp = log_post(x)

    chain = np.empty((n_samples, x.shape[0]))
    logps = np.empty(n_samples)
    n_acc = 0
    for k in range(n_samples):
        x, logp, accepted = mh_step(log_post, x, logp, cov, rng=rng)
        chain[k] = x
        logps[k] = logp
        n_acc += bool(accepted)

    return _finish(chain, logps, param_names, n_acc / n_samples)


def ram(log_post, x0, n_samples, initial_cov=None, gamma=0.51,
        target_rate=0.234, param_names=None, rng=None):
    """Run Robust Adaptive Metropolis to completion.

    Self-tunes the proposal covariance, so a rough ``initial_cov`` is fine.

    Returns
    -------
    Result
    """
    log_post, param_names = _resolve(log_post, param_names)
    x = np.asarray(x0, dtype=float)
    d = x.shape[0]
    S = np.linalg.cholesky(as_cov(0.1**2 if initial_cov is None else initial_cov, d))
    logp = log_post(x)

    chain = np.empty((n_samples, d))
    logps = np.empty(n_samples)
    n_acc = 0
    for k in range(n_samples):
        x, logp, S, accepted = ram_step(
            log_post, x, logp, S, k + 1, gamma=gamma,
            target_rate=target_rate, rng=rng,
        )
        chain[k] = x
        logps[k] = logp
        n_acc += bool(accepted)

    return _finish(chain, logps, param_names, n_acc / n_samples)


def mala(log_post_and_grad, x0, n_samples, step_size, param_names=None, rng=None):
    """Run MALA to completion.

    Parameters
    ----------
    log_post_and_grad : callable or Problem
        ``theta -> (log posterior, gradient)``, or a Problem carrying
        gradients.

    Returns
    -------
    Result
    """
    f, param_names = _resolve_grad(log_post_and_grad, param_names)
    x = np.asarray(x0, dtype=float)
    logp, grad = f(x)

    chain = np.empty((n_samples, x.shape[0]))
    logps = np.empty(n_samples)
    n_acc = 0
    for k in range(n_samples):
        x, logp, grad, accepted = mala_step(f, x, logp, grad, step_size, rng=rng)
        chain[k] = x
        logps[k] = logp
        n_acc += bool(accepted)

    return _finish(chain, logps, param_names, n_acc / n_samples)


def adaptive_mala(log_post_and_grad, x0, n_samples, initial_step_size=0.1,
                  gamma=0.51, target_rate=0.574, param_names=None, rng=None):
    """Run MALA with log-space step-size adaptation to completion.

    Returns
    -------
    Result
    """
    f, param_names = _resolve_grad(log_post_and_grad, param_names)
    x = np.asarray(x0, dtype=float)
    logp, grad = f(x)
    log_step = np.log(initial_step_size)

    chain = np.empty((n_samples, x.shape[0]))
    logps = np.empty(n_samples)
    n_acc = 0
    for k in range(n_samples):
        x, logp, grad, log_step, accepted = adaptive_mala_step(
            f, x, logp, grad, log_step, k + 1,
            gamma=gamma, target_rate=target_rate, rng=rng,
        )
        chain[k] = x
        logps[k] = logp
        n_acc += bool(accepted)

    return _finish(chain, logps, param_names, n_acc / n_samples)


def dram(log_post, x0, n_samples, initial_cov=None, dr_scale=0.1,
         adapt_start=100, adapt_interval=10, regularization=1e-6,
         param_names=None, rng=None):
    """Run Delayed Rejection Adaptive Metropolis to completion.

    Returns
    -------
    Result
    """
    log_post, param_names = _resolve(log_post, param_names)
    x = np.asarray(x0, dtype=float)
    state = init_dram_state(x, initial_cov)
    logp = log_post(x)

    chain = np.empty((n_samples, x.shape[0]))
    logps = np.empty(n_samples)
    n_acc = 0
    for k in range(n_samples):
        x, logp, state, accepted = dram_step(
            log_post, x, logp, state, dr_scale=dr_scale,
            adapt_start=adapt_start, adapt_interval=adapt_interval,
            regularization=regularization, rng=rng,
        )
        chain[k] = x
        logps[k] = logp
        n_acc += bool(accepted)

    return _finish(chain, logps, param_names, n_acc / n_samples)


def gibbs(log_post, x0, n_samples, blocks=None, proposal_std=0.5,
          param_names=None, rng=None):
    """Run Metropolis-within-Gibbs to completion.

    Parameters
    ----------
    blocks : sequence of sequence of int, optional
        Index groups updated together. Defaults to one block per parameter.

    Returns
    -------
    Result
        ``acceptance_rate`` is the mean over blocks. Per-block rates are on
        the returned object as ``block_acceptance_rates``.
    """
    log_post, param_names = _resolve(log_post, param_names)
    x = np.asarray(x0, dtype=float)
    d = x.shape[0]
    if blocks is None:
        blocks = [[i] for i in range(d)]
    logp = log_post(x)

    chain = np.empty((n_samples, d))
    logps = np.empty(n_samples)
    n_acc = np.zeros(len(blocks))
    for k in range(n_samples):
        x, logp, accepted = gibbs_step(
            log_post, x, logp, blocks, proposal_std, rng=rng
        )
        chain[k] = x
        logps[k] = logp
        n_acc += np.asarray(accepted)

    rates = n_acc / n_samples
    result = _finish(chain, logps, param_names, float(rates.mean()))
    result.block_acceptance_rates = rates
    return result
