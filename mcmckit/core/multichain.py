"""Multi-chain MCMC execution and the MultiChainResult container."""

from __future__ import annotations

import numpy as np

from .parallel import WorkerPool, resolve_n_workers
from .result import Result
from .diagnostics import gelman_rubin, ess, convergence_summary


class MultiChainResult:
    """Container for posterior samples from multiple independent chains.

    Produced by :func:`run_chains`.  Provides convergence diagnostics
    (:meth:`gelman_rubin`, :meth:`ess`, :meth:`summary`) and can pool all
    chains into a single :class:`~mcmckit.core.result.Result` for downstream
    analysis.

    Parameters
    ----------
    results : list of Result
        One :class:`~mcmckit.core.result.Result` per chain.

    Examples
    --------
    ::

        mc_result = mc.run_chains(
            mc.DRAM(n_samples=10_000, initial_cov=np.eye(2)),
            problem,
            x0=[1.0, 1.0],
            n_chains=4,
        )
        mc_result.summary()
        pooled = mc_result.pool(discard=2000)
        pooled.plot_corner()
    """

    def __init__(self, results: list):
        if len(results) < 1:
            raise ValueError("MultiChainResult requires at least one chain.")
        self._results = list(results)

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, j) -> Result:
        return self._results[j]

    def __iter__(self):
        return iter(self._results)

    def __repr__(self) -> str:
        n = len(self._results)
        s = self._results[0].samples.shape[0]
        d = self._results[0].samples.shape[1]
        return f"MultiChainResult(n_chains={n}, n_samples={s}, n_params={d})"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def gelman_rubin(self) -> np.ndarray:
        """Gelman-Rubin :math:`\\hat{R}` per parameter (split-chain).

        Returns
        -------
        np.ndarray, shape (n_params,)
            Values near 1.0 indicate convergence.  Common threshold: < 1.01.
        """
        return gelman_rubin(self._results)

    def ess(self) -> np.ndarray:
        """Per-parameter ESS for each chain, shape (n_chains, n_params).

        Returns
        -------
        np.ndarray, shape (n_chains, n_params)
        """
        return np.array([ess(r.samples) for r in self._results])

    def summary(self, discard: int = 0, threshold_rhat: float = 1.01,
                print_table: bool = True) -> dict:
        """Print a convergence summary table and return diagnostics.

        Parameters
        ----------
        discard : int
            Burn-in samples to discard before computing diagnostics.
        threshold_rhat : float
            Warn if :math:`\\hat{R}` exceeds this value.
        print_table : bool
            Whether to print to stdout.

        Returns
        -------
        dict with keys ``rhat``, ``ess``, ``converged``, ``warnings``.
        """
        chains = [r.discard(discard) if discard > 0 else r
                  for r in self._results]
        diag = convergence_summary(chains, threshold_rhat=threshold_rhat)

        if print_table:
            d = self._results[0].samples.shape[1]
            names = self._results[0].param_names or [f"theta[{i}]" for i in range(d)]
            ess_arr = np.array([ess(c.samples) for c in chains])  # (n_chains, d)
            total_ess = ess_arr.sum(axis=0)

            header = (f"{'Parameter':<18}  {'R-hat':>7}  {'ESS (total)':>12}  "
                      f"{'Mean':>10}  {'Std':>10}")
            print(header)
            print("-" * len(header))
            pooled_samples = np.concatenate([c.samples for c in chains], axis=0)
            for i, name in enumerate(names):
                rhat_i = diag["rhat"][i]
                flag = " *" if (not np.isnan(rhat_i) and rhat_i > threshold_rhat) else ""
                print(f"{name:<18}  {rhat_i:>7.4f}  {total_ess[i]:>12.0f}  "
                      f"{pooled_samples[:, i].mean():>10.4f}  "
                      f"{pooled_samples[:, i].std():>10.4f}{flag}")

            if diag["warnings"]:
                print()
                for w in diag["warnings"]:
                    print(f"  WARNING: {w}")
            else:
                print(f"\n  All R-hat < {threshold_rhat} — chains appear converged.")

        return diag

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def pool(self, discard: int = 0) -> Result:
        """Concatenate all chains into a single Result.

        Parameters
        ----------
        discard : int
            Number of burn-in samples to drop from the start of each chain
            before pooling.

        Returns
        -------
        Result
        """
        chains = [r.discard(discard) if discard > 0 else r
                  for r in self._results]
        samples = np.concatenate([c.samples for c in chains], axis=0)
        log_posts = np.concatenate([c.log_posteriors for c in chains], axis=0)
        return Result(
            samples=samples,
            log_posteriors=log_posts,
            param_names=self._results[0].param_names,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_traces(self, title: str | None = None):
        """Overlaid trace plots for all chains.

        Each chain is drawn in a different colour.

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        d = self._results[0].samples.shape[1]
        names = self._results[0].param_names or [f"theta[{i}]" for i in range(d)]
        n_chains = len(self._results)
        cmap = plt.cm.tab10

        fig, axes = plt.subplots(d, 1, figsize=(10, 2.5 * d), squeeze=False,
                                 constrained_layout=True)
        for i, ax in enumerate(axes[:, 0]):
            for j, res in enumerate(self._results):
                ax.plot(res.samples[:, i], lw=0.6, alpha=0.8,
                        color=cmap(j / max(n_chains, 10)),
                        label=f"chain {j}")
            ax.set_ylabel(names[i])
        axes[-1, 0].set_xlabel("iteration")
        axes[0, 0].legend(fontsize=7, ncol=n_chains, loc="upper right")

        if title is not None:
            fig.suptitle(title)
        return fig


# ------------------------------------------------------------------
# run_chains
# ------------------------------------------------------------------

def _run_one_chain(args):
    """Run a single chain. Module level so that process workers can pickle it."""
    sampler, problem, start = args
    return sampler.run(problem, x0=start)


def run_chains(
    sampler,
    problem,
    x0,
    n_chains: int = 4,
    jitter_scale: float = 0.1,
    n_workers: int = 1,
    backend: str = "auto",
) -> MultiChainResult:
    """Run multiple independent chains and return a :class:`MultiChainResult`.

    Parameters
    ----------
    sampler
        An mcmckit sampler instance (e.g. :class:`~mcmckit.samplers.dram.DRAM`).
        A **fresh copy** is made for each chain so that internal adaptation
        state does not leak between chains.
    problem
        A :class:`~mcmckit.core.problem.Problem` or
        :class:`~mcmckit.core.hierarchical.HierarchicalProblem`.
    x0 : array-like or list of array-like
        Starting point(s).

        * **Single vector** — a 1-D array or a flat list of scalars such as
          ``[10.0, 8.0]``.  All chains start near ``x0`` with a small random
          jitter of ``jitter_scale * |x0|`` (or ``jitter_scale`` when the
          component is zero).
        * **List of vectors** — a list of arrays/lists, e.g.
          ``[[10.0, 8.0], [9.5, 7.5], ...]``, one per chain.  Each chain
          starts exactly at the corresponding point; ``n_chains`` is inferred
          from the list length.

    n_chains : int
        Number of chains.  Ignored when ``x0`` is a list of vectors.
    jitter_scale : float
        Relative jitter applied to a single ``x0``.  Default 0.1 (10%).
    n_workers : int
        Number of chains to run at once.  Default 1 (serial, unchanged
        behaviour).  ``-1`` uses one worker per core.  Chains are independent,
        so this is the cheapest parallelism in the package.
    backend : str
        ``'process'``, ``'thread'`` or ``'auto'``.  See
        :mod:`mcmckit.core.parallel`.  Process workers need a picklable
        problem, which rules out lambdas and closures; ``'auto'`` falls back to
        threads when that is not the case.

        .. note::
           With ``backend='process'`` each worker seeds its own random stream,
           so results are **not** reproducible from ``np.random.seed`` the way
           the serial path is.  Use ``n_workers=1`` when you need bit-for-bit
           reproducibility.

    Returns
    -------
    MultiChainResult

    Examples
    --------
    ::

        sampler = mc.DRAM(n_samples=10_000, initial_cov=np.eye(d))
        mc_result = mc.run_chains(sampler, problem, x0=np.zeros(d), n_chains=4)
        mc_result.summary(discard=2000)
        pooled = mc_result.pool(discard=2000)
    """
    import copy

    # Determine whether x0 is a single starting point or a list of them.
    # A list of lists/arrays → multiple explicit starting points.
    # A list of scalars or a numpy array → single starting point to jitter.
    _is_multi = (
        isinstance(x0, list)
        and len(x0) > 0
        and isinstance(x0[0], (list, np.ndarray))
    )

    # Build list of starting points
    if _is_multi:
        starts = [np.asarray(s, dtype=float) for s in x0]
        n_chains = len(starts)
    else:
        x0 = np.asarray(x0, dtype=float)
        starts = []
        for _ in range(n_chains):
            scale = np.where(np.abs(x0) > 0,
                             jitter_scale * np.abs(x0),
                             jitter_scale)
            starts.append(x0 + np.random.normal(0, scale, size=x0.shape))

    # A fresh copy per chain so adaptation state cannot leak between them.
    jobs = [(copy.deepcopy(sampler), problem, start) for start in starts]

    n_workers = resolve_n_workers(n_workers)
    if n_workers > 1:
        n_workers = min(n_workers, len(jobs))     # no point in idle workers

    with WorkerPool(
        n_workers=n_workers, backend=backend, func=problem.log_likelihood
    ) as pool:
        results = pool.map(_run_one_chain, jobs)

    return MultiChainResult(results)
