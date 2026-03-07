import numpy as np

from .base import BaseSampler
from ..core.result import Result


class AdaptiveMALA(BaseSampler):
    """Adaptive MALA: MALA with automatic step-size tuning.

    Uses the same gradient-biased proposal as MALA but adapts the scalar
    step size ε during sampling to drive the acceptance rate toward the
    theoretically optimal target of 0.574 for MALA.

    Adaptation rule (log-scale stochastic approximation):

        log(ε_{i+1}) = log(ε_i) + η_i (α_i - α*)

    where η_i = (i+1)^{-γ} is a decaying schedule and α* = 0.574.

    Parameters
    ----------
    n_samples : int
        Number of samples to collect when calling run().
    initial_step_size : float
        Starting value for ε. The adaptation will correct a bad initial
        guess, but a reasonable starting point (e.g. 0.1–1.0) helps.
    target_rate : float
        Target acceptance rate α*. Default 0.574 (optimal for MALA).
    gamma : float
        Decay exponent for the adaptation schedule η_i = (i+1)^{-γ}.
        Must be in (0.5, 1]. Default 0.6.

    Notes
    -----
    Requires ``problem.has_grad == True``.
    """

    def __init__(self, n_samples, initial_step_size=0.1, target_rate=0.574, gamma=0.6):
        if not (0.5 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0.5, 1], got {gamma}")
        self.n_samples = n_samples
        self.initial_step_size = initial_step_size
        self.target_rate = target_rate
        self.gamma = gamma
        self._initialized = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self, problem, x0):
        if not problem.has_grad:
            raise RuntimeError(
                "AdaptiveMALA requires grad_log_likelihood and grad_log_prior "
                "to be provided to Problem."
            )
        x0 = np.asarray(x0, dtype=float)
        self._problem = problem
        self.current = x0.copy()
        self.current_logp, self.current_grad = problem.log_posterior_and_grad(self.current)
        self._log_step = np.log(self.initial_step_size)

        self._samples: list[np.ndarray] = []
        self._log_posteriors: list[float] = []
        self._n_accepted = 0
        self._n_steps = 0
        self._initialized = True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def step(self):
        """Perform one adaptive MALA step."""
        if not self._initialized:
            raise RuntimeError("Call initialize(problem, x0) before step().")

        eps = np.exp(self._log_step)
        d = len(self.current)

        # MALA proposal
        mean_fwd = self.current + 0.5 * eps**2 * self.current_grad
        r = np.random.randn(d)
        proposal = mean_fwd + eps * r

        logp_prop, grad_prop = self._problem.log_posterior_and_grad(proposal)

        # Asymmetric proposal correction (same as MALA)
        log_q_fwd = self._log_proposal_density(self.current, proposal, self.current_grad, eps)
        log_q_rev = self._log_proposal_density(proposal, self.current, grad_prop, eps)
        log_alpha = logp_prop + log_q_rev - self.current_logp - log_q_fwd
        alpha = min(1.0, np.exp(log_alpha))

        # Adapt step size in log space
        i = self._n_steps + 1
        eta = i ** (-self.gamma)
        self._log_step += eta * (alpha - self.target_rate)

        # Accept / reject
        if np.random.rand() < alpha:
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

    def _log_proposal_density(self, x_from, x_to, grad_from, eps):
        mean = x_from + 0.5 * eps**2 * grad_from
        diff = x_to - mean
        return -0.5 / (eps**2) * np.dot(diff, diff)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step_size(self):
        """Current (adapted) step size ε."""
        return np.exp(self._log_step)

    @property
    def acceptance_rate(self):
        if self._n_steps == 0:
            return None
        return self._n_accepted / self._n_steps

    @property
    def n_steps(self):
        return self._n_steps
