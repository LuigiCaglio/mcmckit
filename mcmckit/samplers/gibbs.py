import numpy as np

from .base import BaseSampler
from ..core.result import Result


class Gibbs(BaseSampler):
    """Metropolis-within-Gibbs sampler.

    Updates parameter blocks sequentially: at each step, one block is proposed
    while all other parameters are held fixed. The acceptance uses the full
    log-posterior, preserving the correct joint distribution.

    This is the appropriate Gibbs implementation for black-box posteriors where
    the conditional distributions p(θ_block | θ_rest, y) cannot be sampled
    directly.

    Parameters
    ----------
    n_samples : int
        Number of full sweeps (one sweep = one update per block).
    blocks : list of list of int, optional
        Parameter index groups to update together.
        E.g. ``[[0, 1], [2, 3, 4]]`` updates params 0,1 jointly, then 2,3,4.
        Default: one parameter per block (scalar MH for each dimension).
    proposal_std : float or list of float
        Standard deviation of the Gaussian proposal for each block.
        If a scalar, the same std is used for all blocks.
        If a list, must match the number of blocks.

    Examples
    --------
    # Scalar Gibbs (one parameter at a time)
    sampler = Gibbs(n_samples=5000, proposal_std=0.3)
    result = sampler.run(problem, x0=[0.0, 0.0])

    # Block Gibbs
    sampler = Gibbs(n_samples=5000, blocks=[[0, 1], [2, 3]], proposal_std=[0.3, 0.1])
    result = sampler.run(problem, x0=[0.0, 0.0, 0.0, 0.0])
    """

    def __init__(self, n_samples, blocks=None, proposal_std=0.1):
        self.n_samples = n_samples
        self._blocks_spec = blocks
        self._proposal_std_spec = proposal_std
        self._initialized = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self, problem, x0):
        x0 = np.asarray(x0, dtype=float)
        d = x0.shape[0]

        # Build block list
        if self._blocks_spec is None:
            blocks = [[i] for i in range(d)]
        else:
            blocks = [list(b) for b in self._blocks_spec]

        # Build per-block proposal std
        std_spec = self._proposal_std_spec
        if np.isscalar(std_spec):
            proposal_stds = [float(std_spec)] * len(blocks)
        else:
            proposal_stds = list(std_spec)
            if len(proposal_stds) != len(blocks):
                raise ValueError(
                    f"proposal_std has {len(proposal_stds)} entries but "
                    f"there are {len(blocks)} blocks."
                )

        self._blocks = blocks
        self._proposal_stds = proposal_stds
        self._problem = problem
        self.current = x0.copy()
        self.current_logp = problem.log_posterior(self.current)

        # Per-block acceptance counters
        self._n_accepted = [0] * len(blocks)
        self._n_steps = 0
        self._samples: list[np.ndarray] = []
        self._log_posteriors: list[float] = []
        self._initialized = True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def step(self):
        """One full Gibbs sweep: update every block once."""
        if not self._initialized:
            raise RuntimeError("Call initialize(problem, x0) before step().")

        for k, (block, std) in enumerate(zip(self._blocks, self._proposal_stds)):
            block_size = len(block)

            # Propose new values for this block only
            proposal = self.current.copy()
            proposal[block] += np.random.randn(block_size) * std

            logp_prop = self._problem.log_posterior(proposal)
            log_alpha = logp_prop - self.current_logp

            if np.log(np.random.rand()) < log_alpha:
                self.current = proposal
                self.current_logp = logp_prop
                self._n_accepted[k] += 1

        self._samples.append(self.current.copy())
        self._log_posteriors.append(self.current_logp)
        self._n_steps += 1

    def run(self, problem, x0):
        """Initialize and run for n_samples sweeps, returning a Result."""
        self.initialize(problem, x0)
        for _ in range(self.n_samples):
            self.step()
        return self.get_result()

    def get_result(self):
        """Return a Result from all samples collected so far."""
        if not self._samples:
            raise RuntimeError("No samples collected yet.")
        total_accepted = sum(self._n_accepted)
        total_proposals = self._n_steps * len(self._blocks)
        return Result(
            samples=np.array(self._samples),
            log_posteriors=np.array(self._log_posteriors),
            param_names=self._problem.param_names,
            acceptance_rate=total_accepted / total_proposals if total_proposals > 0 else None,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def block_acceptance_rates(self):
        """Per-block acceptance rates."""
        if self._n_steps == 0:
            return None
        return [n / self._n_steps for n in self._n_accepted]

    @property
    def acceptance_rate(self):
        rates = self.block_acceptance_rates
        if rates is None:
            return None
        return float(np.mean(rates))

    @property
    def n_steps(self):
        return self._n_steps
