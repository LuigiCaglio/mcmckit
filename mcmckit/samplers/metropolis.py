import numpy as np

from .base import BaseSampler
from ..core.result import Result


class MetropolisHastings(BaseSampler):
    """Metropolis-Hastings sampler with a fixed Gaussian proposal.

    Parameters
    ----------
    proposal_cov : array-like
        Covariance matrix for the Gaussian proposal. Shape (d, d) or scalar
        (interpreted as a diagonal with that variance for all dimensions).
    n_samples : int
        Number of samples to collect when calling run().

    Examples
    --------
    # Full run
    sampler = MetropolisHastings(proposal_cov=np.eye(2) * 0.1, n_samples=5000)
    result = sampler.run(problem, x0=[0.0, 0.0])

    # Step-by-step (stop whenever you want)
    sampler = MetropolisHastings(proposal_cov=np.eye(2) * 0.1, n_samples=5000)
    sampler.initialize(problem, x0=[0.0, 0.0])
    for i in range(5000):
        sampler.step()
        if i % 500 == 0:
            print(sampler.current)   # inspect current state
    result = sampler.get_result()
    """

    def __init__(self, proposal_cov, n_samples):
        self.n_samples = n_samples
        self._cov = None          # set in initialize after we know dimensionality
        self._cov_input = proposal_cov
        self._initialized = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self, problem, x0):
        x0 = np.asarray(x0, dtype=float)
        d = x0.shape[0]

        cov = np.asarray(self._cov_input, dtype=float)
        if cov.ndim == 0:
            cov = np.eye(d) * float(cov)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        if cov.shape != (d, d):
            raise ValueError(f"proposal_cov shape {cov.shape} incompatible with x0 dimension {d}")

        self._cov = cov
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
        """Perform one Metropolis-Hastings step, updating internal state."""
        if not self._initialized:
            raise RuntimeError("Call initialize(problem, x0) before step().")

        proposal = np.random.multivariate_normal(self.current, self._cov)
        logp_prop = self._problem.log_posterior(proposal)

        log_alpha = logp_prop - self.current_logp
        if np.log(np.random.rand()) < log_alpha:
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
    def acceptance_rate(self):
        if self._n_steps == 0:
            return None
        return self._n_accepted / self._n_steps

    @property
    def n_steps(self):
        return self._n_steps
