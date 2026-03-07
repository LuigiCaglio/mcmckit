import numpy as np

from .base import BaseSampler
from ..core.result import Result


class MALA(BaseSampler):
    """Metropolis-Adjusted Langevin Algorithm (MALA).

    Uses the gradient of the log-posterior to bias proposals toward higher
    probability regions, giving faster mixing than vanilla MH.

    Proposal:
        x* = x + (ε²/2) * ∇log p(x|y) + ε * noise,   noise ~ N(0, I)

    The MH accept/reject step corrects for the asymmetric proposal, so the
    chain targets the exact posterior.

    Parameters
    ----------
    step_size : float
        Langevin step size ε. Smaller = higher acceptance, slower mixing.
        Larger = lower acceptance, faster exploration when accepted.
        Typical starting point: tune so acceptance rate is ~0.57 (optimal
        for MALA in high dimensions, vs ~0.23 for MH).
    n_samples : int
        Number of samples to collect when calling run().

    Notes
    -----
    Requires ``problem.has_grad == True``, i.e. both ``grad_log_likelihood``
    and ``grad_log_prior`` must be provided to the Problem.

    Examples
    --------
    # Full run
    sampler = MALA(step_size=0.1, n_samples=5000)
    result = sampler.run(problem, x0=[0.0, 0.0])

    # Step-by-step
    sampler = MALA(step_size=0.1, n_samples=5000)
    sampler.initialize(problem, x0=[0.0, 0.0])
    for i in range(5000):
        sampler.step()
    result = sampler.get_result()
    """

    def __init__(self, step_size, n_samples):
        self.step_size = step_size
        self.n_samples = n_samples
        self._initialized = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self, problem, x0):
        if not problem.has_grad:
            raise RuntimeError(
                "MALA requires grad_log_likelihood and grad_log_prior "
                "to be provided to Problem."
            )
        x0 = np.asarray(x0, dtype=float)
        self._problem = problem
        self.current = x0.copy()
        self.current_logp, self.current_grad = problem.log_posterior_and_grad(self.current)

        self._samples: list[np.ndarray] = []
        self._log_posteriors: list[float] = []
        self._n_accepted = 0
        self._n_steps = 0
        self._initialized = True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def step(self):
        """Perform one MALA step, updating internal state."""
        if not self._initialized:
            raise RuntimeError("Call initialize(problem, x0) before step().")

        eps = self.step_size
        d = len(self.current)

        # Forward proposal: drift toward higher probability + noise
        mean_fwd = self.current + 0.5 * eps**2 * self.current_grad
        proposal = mean_fwd + eps * np.random.randn(d)

        logp_prop, grad_prop = self._problem.log_posterior_and_grad(proposal)

        # MH correction for asymmetric proposal
        log_q_fwd = self._log_proposal_density(self.current, proposal, self.current_grad)
        log_q_rev = self._log_proposal_density(proposal, self.current, grad_prop)

        log_alpha = logp_prop + log_q_rev - self.current_logp - log_q_fwd

        if np.log(np.random.rand()) < log_alpha:
            self.current = proposal
            self.current_logp = logp_prop
            self.current_grad = grad_prop
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
    # Helpers
    # ------------------------------------------------------------------

    def _log_proposal_density(self, x_from, x_to, grad_from):
        """Log density of proposing x_to given x_from and its gradient."""
        eps = self.step_size
        mean = x_from + 0.5 * eps**2 * grad_from
        diff = x_to - mean
        return -0.5 / (eps**2) * np.dot(diff, diff)

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
