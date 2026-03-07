import numpy as np

from .base import BaseSampler
from ..core.result import Result


class RAM(BaseSampler):
    """Robust Adaptive Metropolis (RAM) sampler.

    Adapts the Cholesky factor S of the proposal covariance at every step
    to drive the empirical acceptance rate toward a target value, without
    requiring any tuning beyond an initial covariance guess.

    Update rule (Vihola 2012, also Algorithm 16.6 in Sarkka & Svensson 2023):

        S_i S_i^T = S_{i-1} (I + η_i (α_i - α*) r_i r_i^T / ||r_i||²) S_{i-1}^T

    where:
        η_i  = i^{-γ}      (step size schedule, γ ∈ (0.5, 1])
        α_i  = min(1, p(θ*|y) / p(θ|y))   (acceptance probability)
        α*   = target acceptance rate (0.234 is optimal for high-d MH)
        r_i  ~ N(0, I)     (the raw proposal noise)

    Parameters
    ----------
    n_samples : int
        Number of samples to collect when calling run().
    initial_cov : array-like, optional
        Initial proposal covariance. Accepts a scalar (isotropic),
        1D array (diagonal), or 2D array (full matrix).
        Defaults to 0.1² * I if not provided.
    target_rate : float
        Target acceptance rate α*. Default 0.234 (theoretically optimal
        for MH in high dimensions). Use ~0.44 for 1D problems.
    gamma : float
        Decay exponent γ for the adaptation schedule η_i = i^{-γ}.
        Must be in (0.5, 1]. Default 0.51 (fast early adaptation,
        slow late adaptation — close to the lower bound is typical).

    Examples
    --------
    sampler = RAM(n_samples=10_000)
    result = sampler.run(problem, x0=[0.0, 0.0])

    # Step-by-step with live covariance inspection
    sampler = RAM(n_samples=10_000)
    sampler.initialize(problem, x0=[0.0, 0.0])
    for i in range(10_000):
        sampler.step()
        if i % 1000 == 0:
            print(f"step {i} | acc={sampler.acceptance_rate:.3f} | cov diag={np.diag(sampler.proposal_cov)}")
    result = sampler.get_result()

    References
    ----------
    Vihola, M. (2012). Robust adaptive Metropolis algorithm with coerced
    acceptance rate. Statistics and Computing, 22(5), 997-1008.
    """

    def __init__(self, n_samples, initial_cov=None, target_rate=0.234, gamma=0.51):
        if not (0.5 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0.5, 1], got {gamma}")
        self.n_samples = n_samples
        self._initial_cov = initial_cov
        self.target_rate = target_rate
        self.gamma = gamma
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

        self._S = np.linalg.cholesky(cov)
        self._problem = problem
        self.current = x0.copy()
        self.current_logp = problem.log_posterior(self.current)

        self._samples: list[np.ndarray] = []
        self._log_posteriors: list[float] = []
        self._n_accepted = 0
        self._n_steps = 0
        self._initialized = True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def step(self):
        """Perform one RAM step: propose, adapt S, accept/reject."""
        if not self._initialized:
            raise RuntimeError("Call initialize(problem, x0) before step().")

        i = self._n_steps + 1  # 1-indexed to match the algorithm
        d = len(self.current)

        # Draw standard Gaussian noise and form proposal
        r = np.random.randn(d)
        proposal = self.current + self._S @ r
        logp_prop = self._problem.log_posterior(proposal)

        # Acceptance probability
        alpha = min(1.0, np.exp(logp_prop - self.current_logp))

        # Rank-1 Cholesky adaptation
        r_sq = np.dot(r, r)
        if r_sq > 0:
            eta = i ** (-self.gamma)
            rank1 = np.eye(d) + eta * (alpha - self.target_rate) * np.outer(r, r) / r_sq
            SiSiT = self._S @ rank1 @ self._S.T
            try:
                self._S = np.linalg.cholesky(SiSiT)
            except np.linalg.LinAlgError:
                pass  # keep old S if update breaks positive definiteness

        # Accept / reject
        if np.random.rand() < alpha:
            self.current = proposal
            self.current_logp = logp_prop
            self._n_accepted += 1

        self._samples.append(self.current.copy())
        self._log_posteriors.append(self.current_logp)
        self._n_steps += 1

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
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def proposal_cov(self):
        """Current proposal covariance S S^T (adapted during sampling)."""
        return self._S @ self._S.T

    @property
    def proposal_chol(self):
        """Current Cholesky factor S of the proposal covariance."""
        return self._S.copy()

    @property
    def acceptance_rate(self):
        if self._n_steps == 0:
            return None
        return self._n_accepted / self._n_steps

    @property
    def n_steps(self):
        return self._n_steps
