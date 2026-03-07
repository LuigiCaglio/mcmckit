import numpy as np


class Problem:
    """Encapsulates a Bayesian inference problem.

    Parameters
    ----------
    prior : callable
        Function theta -> log p(theta). Must return a scalar float.
    likelihood : callable
        Function theta -> log p(y | theta). Must return a scalar float.
    bounds : list of (lo, hi) tuples, optional
        Hard parameter bounds. Any theta outside bounds returns -inf
        without evaluating the prior or likelihood.
    param_names : list of str, optional
        Names for each parameter dimension. Used in Result plots/summaries.
    grad_log_likelihood : callable, optional
        Function theta -> gradient of log p(y | theta). Returns array of
        shape (d,). Used by gradient-based samplers (MALA, HMC). Standard
        MH ignores this entirely.
    grad_log_prior : callable, optional
        Function theta -> gradient of log p(theta). Returns array of
        shape (d,). Used by gradient-based samplers together with
        grad_log_likelihood to form the full log-posterior gradient.
    """

    def __init__(
        self,
        prior,
        likelihood,
        bounds=None,
        param_names=None,
        grad_log_likelihood=None,
        grad_log_prior=None,
    ):
        self.prior = prior
        self.likelihood = likelihood
        self.bounds = bounds
        self.param_names = param_names
        self.grad_log_likelihood = grad_log_likelihood
        self.grad_log_prior = grad_log_prior

    # ------------------------------------------------------------------
    # Core evaluations
    # ------------------------------------------------------------------

    def log_prior(self, theta):
        if self.bounds is not None:
            for val, (lo, hi) in zip(theta, self.bounds):
                if not (lo <= val <= hi):
                    return -np.inf
        return float(self.prior(theta))

    def log_likelihood(self, theta):
        return float(self.likelihood(theta))

    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return lp
        return lp + self.log_likelihood(theta)

    # ------------------------------------------------------------------
    # Gradient evaluations (only available when callables are provided)
    # ------------------------------------------------------------------

    @property
    def has_grad(self):
        """True if gradients of both prior and likelihood are available."""
        return self.grad_log_likelihood is not None and self.grad_log_prior is not None

    def grad_log_posterior(self, theta):
        """Gradient of log p(theta | y). Requires has_grad == True."""
        if not self.has_grad:
            raise RuntimeError(
                "Gradient requested but grad_log_likelihood / grad_log_prior "
                "were not provided to Problem."
            )
        return np.asarray(self.grad_log_prior(theta)) + np.asarray(self.grad_log_likelihood(theta))

    def log_posterior_and_grad(self, theta):
        """Return (log_posterior, gradient) in a single call."""
        return self.log_posterior(theta), self.grad_log_posterior(theta)
