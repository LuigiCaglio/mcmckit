import numpy as np

from .base import BaseSampler
from ..core.result import Result


class DRAM(BaseSampler):
    """Delayed Rejection Adaptive Metropolis (DRAM) sampler.

    Combines two ideas:

    **Adaptive Metropolis (AM)**: the proposal covariance is updated during
    sampling using the empirical covariance of all past samples, targeting
    the theoretically optimal scaling 2.38²/d.

    **Delayed Rejection (DR)**: when the first proposal is rejected, a second
    (smaller) proposal is attempted instead of immediately staying put. The
    second-stage acceptance criterion accounts for the fact that the first
    proposal was rejected, preserving detailed balance.

    Parameters
    ----------
    n_samples : int
        Number of samples to collect when calling run().
    initial_cov : array-like, optional
        Initial proposal covariance. Scalar, 1D (diagonal), or 2D (full).
        Defaults to 0.1² * I.
    adapt_start : int
        Number of samples to collect before starting covariance adaptation.
        Default 100.
    adapt_interval : int
        Update the proposal covariance every this many steps. Default 10.
    dr_scale : float
        The second-stage proposal uses dr_scale² * C₁. Smaller = more
        conservative second attempt. Default 0.1.
    regularization : float
        Small diagonal regularization added to the empirical covariance to
        keep it positive definite. Default 1e-6.

    References
    ----------
    Haario, H., Laine, M., Mira, A., & Saksman, E. (2006). DRAM: Efficient
    adaptive MCMC. Statistics and Computing, 16(4), 339-354.
    """

    def __init__(
        self,
        n_samples,
        initial_cov=None,
        adapt_start=100,
        adapt_interval=10,
        dr_scale=0.1,
        regularization=1e-6,
    ):
        self.n_samples = n_samples
        self._initial_cov = initial_cov
        self.adapt_start = adapt_start
        self.adapt_interval = adapt_interval
        self.dr_scale = dr_scale
        self.regularization = regularization
        self._initialized = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self, problem, x0):
        x0 = np.asarray(x0, dtype=float)
        d = x0.shape[0]

        cov = self._initial_cov
        if cov is None:
            cov = np.eye(d) * 0.1**2
        else:
            cov = np.asarray(cov, dtype=float)
            if cov.ndim == 0:
                cov = np.eye(d) * float(cov)
            elif cov.ndim == 1:
                cov = np.diag(cov)
        if cov.shape != (d, d):
            raise ValueError(f"initial_cov shape {cov.shape} incompatible with x0 dimension {d}")

        self._C = cov.copy()                 # current proposal covariance
        self._sd = 2.38**2 / d              # optimal AM scaling
        self._problem = problem
        self.current = x0.copy()
        self.current_logp = problem.log_posterior(self.current)

        self._samples: list[np.ndarray] = []
        self._log_posteriors: list[float] = []
        self._n_accepted_stage1 = 0
        self._n_accepted_stage2 = 0
        self._n_steps = 0
        self._initialized = True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def step(self):
        """Perform one DRAM step (two-stage delayed rejection + adaptation)."""
        if not self._initialized:
            raise RuntimeError("Call initialize(problem, x0) before step().")

        # ---- Stage 1 proposal ----------------------------------------
        theta1 = np.random.multivariate_normal(self.current, self._C)
        logp1 = self._problem.log_posterior(theta1)

        log_alpha1 = logp1 - self.current_logp
        alpha1 = min(1.0, np.exp(log_alpha1))

        if np.random.rand() < alpha1:
            self.current = theta1
            self.current_logp = logp1
            self._n_accepted_stage1 += 1
        else:
            # ---- Stage 2 proposal ------------------------------------
            C2 = self._C * self.dr_scale**2
            theta2 = np.random.multivariate_normal(self.current, C2)
            logp2 = self._problem.log_posterior(theta2)

            # α₁'(θ₂, θ₁): acceptance prob of θ₁ if we were at θ₂
            log_alpha1_prime = logp1 - logp2
            alpha1_prime = min(1.0, np.exp(log_alpha1_prime))

            # Log proposal ratio: log q₁(θ₁|θ₂) - log q₁(θ₁|θ)
            # q₁ = N(mean, C), so log_ratio = -½[(θ₁-θ₂)ᵀC⁻¹(θ₁-θ₂) - (θ₁-θ)ᵀC⁻¹(θ₁-θ)]
            log_prop_ratio = self._log_proposal_ratio(theta1, theta2, self.current)

            log_1_minus_a1_curr = np.log(max(1.0 - alpha1, 1e-300))
            log_1_minus_a1_prop = np.log(max(1.0 - alpha1_prime, 1e-300))

            log_alpha2 = (logp2 - self.current_logp
                          + log_prop_ratio
                          + log_1_minus_a1_prop - log_1_minus_a1_curr)

            if np.log(np.random.rand()) < log_alpha2:
                self.current = theta2
                self.current_logp = logp2
                self._n_accepted_stage2 += 1

        self._samples.append(self.current.copy())
        self._log_posteriors.append(self.current_logp)
        self._n_steps += 1

        # ---- AM covariance adaptation --------------------------------
        n = self._n_steps
        if n >= self.adapt_start and n % self.adapt_interval == 0:
            samples_so_far = np.array(self._samples)
            emp_cov = np.cov(samples_so_far.T)
            if emp_cov.ndim == 0:
                emp_cov = np.array([[float(emp_cov)]])
            d = len(self.current)
            self._C = self._sd * emp_cov + self.regularization * np.eye(d)

    def run(self, problem, x0):
        """Initialize and run for n_samples steps, returning a Result."""
        self.initialize(problem, x0)
        for _ in range(self.n_samples):
            self.step()
        return self.get_result()

    def get_result(self):
        """Return a Result from all samples collected so far."""
        if not self._samples:
            raise RuntimeError("No samples collected yet.")
        return Result(
            samples=np.array(self._samples),
            log_posteriors=np.array(self._log_posteriors),
            param_names=self._problem.param_names,
            acceptance_rate=self._n_accepted / self._n_steps,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_proposal_ratio(self, theta1, theta2, theta_curr):
        """log q₁(θ₁|θ₂) - log q₁(θ₁|θ_curr) for symmetric Gaussian q₁(·, C)."""
        try:
            L = np.linalg.cholesky(self._C)
            diff2 = np.linalg.solve(L, theta1 - theta2)
            diff_curr = np.linalg.solve(L, theta1 - theta_curr)
            return -0.5 * (np.dot(diff2, diff2) - np.dot(diff_curr, diff_curr))
        except np.linalg.LinAlgError:
            return 0.0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def _n_accepted(self):
        return self._n_accepted_stage1 + self._n_accepted_stage2

    @property
    def acceptance_rate(self):
        if self._n_steps == 0:
            return None
        return self._n_accepted / self._n_steps

    @property
    def stage1_acceptance_rate(self):
        if self._n_steps == 0:
            return None
        return self._n_accepted_stage1 / self._n_steps

    @property
    def stage2_acceptance_rate(self):
        if self._n_steps == 0:
            return None
        return self._n_accepted_stage2 / self._n_steps

    @property
    def proposal_cov(self):
        """Current (adapted) proposal covariance."""
        return self._C.copy()

    @property
    def n_steps(self):
        return self._n_steps
