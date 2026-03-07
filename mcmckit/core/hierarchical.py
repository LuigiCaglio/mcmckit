import numpy as np

from .problem import Problem
from .result import Result


class HierarchicalProblem:
    """Joint Bayesian inference over hyperparameters and group-level parameters.

    Models the situation where you have J groups (e.g. J nominally identical
    structures) each with local parameters :math:`\\theta_j`, linked through
    shared hyperparameters :math:`\\phi`:

    .. math::

        p(\\phi) \\quad\\text{(hyperprior)}

        p(\\theta_j \\mid \\phi) \\quad\\text{(group-level prior)}

        p(y_j \\mid \\theta_j) \\quad\\text{(per-group likelihood)}

    The joint log-posterior is:

    .. math::

        \\log p(\\phi, \\theta_1,\\ldots,\\theta_J \\mid y)
          = \\log p(\\phi)
          + \\sum_j \\log p(\\theta_j \\mid \\phi)
          + \\sum_j \\log p(y_j \\mid \\theta_j)

    The full parameter vector is laid out as:

    .. code-block::

        theta = [phi_0, ..., phi_{n_hyper-1},
                 theta_1_0, ..., theta_1_{n_group-1},
                 ...
                 theta_J_0, ..., theta_J_{n_group-1}]

    This object exposes the same interface as :class:`Problem` and is
    therefore compatible with all samplers.

    Parameters
    ----------
    hyperprior : callable
        ``log p(phi) -> float``.  ``phi`` has shape ``(n_hyper,)``.
    group_prior : callable
        ``log p(theta_j | phi) -> float``.  Called as
        ``group_prior(theta_j, phi)``.
    group_likelihoods : list of callable
        One callable per group: ``log p(y_j | theta_j) -> float``.
        If a single callable is given it is broadcast to all J groups.
    n_hyper : int
        Dimension of ``phi``.
    n_group : int
        Dimension of each group's local parameter vector ``theta_j``.
    param_names_hyper : list of str, optional
        Names for the hyperparameters.
    param_names_group : list of str, optional
        Names for one group's parameters; automatically suffixed with the
        group index, e.g. ``["k[0]", "k[1]", ...]``.

    Examples
    --------
    5 structures sharing a stiffness population::

        def hyperprior(phi):
            mu, log_s = phi
            return -0.5 * ((mu - 10) / 3)**2 - 0.5 * log_s**2

        def group_prior(theta_j, phi):
            mu, log_s = phi
            sigma = np.exp(log_s)
            return -0.5 * ((theta_j[0] - mu) / sigma)**2 - log_s

        likelihoods = [make_likelihood(y_j) for y_j in datasets]

        hproblem = mc.HierarchicalProblem(
            hyperprior=hyperprior,
            group_prior=group_prior,
            group_likelihoods=likelihoods,
            n_hyper=2,
            n_group=1,
            param_names_hyper=["mu_k", "log_sigma_k"],
            param_names_group=["k"],
        )

        result = mc.DRAM(n_samples=20_000, initial_cov=np.eye(hproblem.n_params)).run(
            hproblem, x0=hproblem.default_x0([10.0, np.log(1.0)],
                                              [[10.0]] * 5))

        hyper_result  = hproblem.extract_hyper(result)
        group_results = [hproblem.extract_group(result, j) for j in range(5)]
    """

    def __init__(self, hyperprior, group_prior, group_likelihoods,
                 n_hyper, n_group,
                 param_names_hyper=None, param_names_group=None):

        self._hyperprior = hyperprior
        self._group_prior = group_prior

        # Accept a single likelihood shared by all groups
        if callable(group_likelihoods) and not isinstance(group_likelihoods, list):
            raise ValueError(
                "group_likelihoods must be a list of callables, one per group."
            )
        self._group_likelihoods = list(group_likelihoods)
        self.n_groups = len(self._group_likelihoods)
        self.n_hyper = n_hyper
        self.n_group = n_group

        # Build param_names
        hyper_names = (list(param_names_hyper)
                       if param_names_hyper is not None
                       else [f"phi[{i}]" for i in range(n_hyper)])
        if param_names_group is not None:
            base = list(param_names_group)
        else:
            base = [f"theta[{i}]" for i in range(n_group)]

        group_names = [f"{name}_{j}"
                       for j in range(self.n_groups)
                       for name in base]
        self.param_names = hyper_names + group_names

    # ------------------------------------------------------------------
    # Helpers: split / assemble the full parameter vector
    # ------------------------------------------------------------------

    @property
    def n_params(self):
        """Total number of parameters: n_hyper + n_groups * n_group."""
        return self.n_hyper + self.n_groups * self.n_group

    def split(self, theta):
        """Split full theta into (phi, [theta_0, ..., theta_{J-1}]).

        Parameters
        ----------
        theta : array-like, shape (n_params,)

        Returns
        -------
        phi : np.ndarray, shape (n_hyper,)
        group_thetas : list of np.ndarray, each shape (n_group,)
        """
        theta = np.asarray(theta, dtype=float)
        phi = theta[:self.n_hyper]
        group_thetas = [
            theta[self.n_hyper + j * self.n_group:
                  self.n_hyper + (j + 1) * self.n_group]
            for j in range(self.n_groups)
        ]
        return phi, group_thetas

    def default_x0(self, phi0, group_x0s):
        """Assemble a starting point from hyper and group initial values.

        Parameters
        ----------
        phi0 : array-like, shape (n_hyper,)
        group_x0s : list of array-like, each shape (n_group,)
            One entry per group.

        Returns
        -------
        np.ndarray, shape (n_params,)
        """
        parts = [np.asarray(phi0, dtype=float)]
        for x in group_x0s:
            parts.append(np.asarray(x, dtype=float))
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Problem interface (compatible with all samplers)
    # ------------------------------------------------------------------

    def log_prior(self, theta):
        """log p(phi) + sum_j log p(theta_j | phi)."""
        phi, group_thetas = self.split(theta)
        lp = float(self._hyperprior(phi))
        if not np.isfinite(lp):
            return -np.inf
        for theta_j in group_thetas:
            lp += float(self._group_prior(theta_j, phi))
            if not np.isfinite(lp):
                return -np.inf
        return lp

    def log_likelihood(self, theta):
        """sum_j log p(y_j | theta_j)."""
        _, group_thetas = self.split(theta)
        ll = 0.0
        for theta_j, lik_j in zip(group_thetas, self._group_likelihoods):
            ll += float(lik_j(theta_j))
            if not np.isfinite(ll):
                return -np.inf
        return ll

    def log_posterior(self, theta):
        """log p(phi, theta_1,...,theta_J | y)."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    # ------------------------------------------------------------------
    # Extract marginal Results after sampling
    # ------------------------------------------------------------------

    def extract_hyper(self, result):
        """Return a Result containing only the hyperparameter samples.

        Parameters
        ----------
        result : Result
            Full posterior result from running a sampler on this problem.

        Returns
        -------
        Result, shape (n_samples, n_hyper)
        """
        names = self.param_names[:self.n_hyper]
        return Result(
            samples=result.samples[:, :self.n_hyper],
            log_posteriors=result.log_posteriors,
            param_names=names,
        )

    def extract_group(self, result, j):
        """Return a Result containing only group j's parameter samples.

        Parameters
        ----------
        result : Result
        j : int
            Group index (0-based).

        Returns
        -------
        Result, shape (n_samples, n_group)
        """
        if j < 0 or j >= self.n_groups:
            raise ValueError(f"j must be in [0, {self.n_groups - 1}], got {j}.")
        start = self.n_hyper + j * self.n_group
        end   = start + self.n_group
        # recover group param names from full param_names
        names = self.param_names[start:end]
        return Result(
            samples=result.samples[:, start:end],
            log_posteriors=result.log_posteriors,
            param_names=names,
        )
