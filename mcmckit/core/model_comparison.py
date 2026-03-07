from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Jeffreys scale
# ------------------------------------------------------------------

# Thresholds on |log10(BF)| for the Jeffreys (1961) evidence scale.
_JEFFREYS_THRESHOLDS = [
    (2.0, "Decisive"),
    (1.0, "Strong"),
    (0.5, "Substantial"),
    (0.0, "Barely worth mentioning"),
]


def _jeffreys_label(log10_bf: float) -> str:
    """Map |log10(BF)| to a Jeffreys-scale label."""
    abs_val = abs(log10_bf)
    for threshold, label in _JEFFREYS_THRESHOLDS:
        if abs_val >= threshold:
            return label
    return "Barely worth mentioning"


# ------------------------------------------------------------------
# Bayes factor utility
# ------------------------------------------------------------------

def bayes_factor(log_ev_1: float, log_ev_2: float) -> dict:
    """Compute the Bayes factor B12 = p(y|M1) / p(y|M2).

    The Bayes factor is the ratio of marginal likelihoods (evidences) of two
    competing models.  It is the factor by which the data updates the prior
    odds in favour of M1 over M2:

    .. math::

        B_{12} = \\frac{p(y \\mid M_1)}{p(y \\mid M_2)}
               = \\exp(\\log p(y \\mid M_1) - \\log p(y \\mid M_2))

    Both log-evidences are typically obtained from
    :class:`~mcmckit.samplers.tmcmc.TMCMC` via ``result.log_evidence``.

    Parameters
    ----------
    log_ev_1 : float
        Log marginal likelihood of model M1.
    log_ev_2 : float
        Log marginal likelihood of model M2 (reference model).

    Returns
    -------
    dict with keys:

    ``log_bf`` : float
        Natural-log Bayes factor ln(B12).
    ``log10_bf`` : float
        Base-10 log Bayes factor log10(B12).
    ``bf`` : float
        Bayes factor B12 (capped at 1e300 to avoid overflow).
    ``preferred`` : str
        ``"M1"`` if B12 > 1, ``"M2"`` otherwise.
    ``evidence`` : str
        Jeffreys-scale label (``"Decisive"``, ``"Strong"``,
        ``"Substantial"``, or ``"Barely worth mentioning"``).

    Examples
    --------
    >>> res1 = tmcmc.run(problem1, prior_samples1)
    >>> res2 = tmcmc.run(problem2, prior_samples2)
    >>> mc.bayes_factor(res1.log_evidence, res2.log_evidence)
    {'log_bf': 4.6, 'log10_bf': 2.0, 'bf': 99.5, 'preferred': 'M1',
     'evidence': 'Decisive'}
    """
    log_bf = float(log_ev_1) - float(log_ev_2)
    log10_bf = log_bf / np.log(10)
    bf = np.exp(np.clip(log_bf, -700, 700))
    preferred = "M1" if log_bf > 0 else "M2"
    evidence = _jeffreys_label(log10_bf)
    return dict(
        log_bf=log_bf,
        log10_bf=log10_bf,
        bf=bf,
        preferred=preferred,
        evidence=evidence,
    )


# ------------------------------------------------------------------
# ModelComparison
# ------------------------------------------------------------------

class ModelComparison:
    """Run TMCMC on multiple competing models and compare evidences.

    Automates the workflow:

    1. Run :class:`~mcmckit.samplers.tmcmc.TMCMC` on each model to obtain the
       log marginal likelihood (log-evidence).
    2. Rank models by evidence.
    3. Compute Bayes factors relative to the best model.
    4. Report a summary table with Jeffreys-scale interpretation.

    Parameters
    ----------
    models : list of tuple
        Each entry is ``(name, problem, prior_samples)`` where

        * ``name`` – a short string label for the model
        * ``problem`` – a :class:`~mcmckit.core.problem.Problem` (or
          :class:`~mcmckit.core.hierarchical.HierarchicalProblem`) instance
        * ``prior_samples`` – ``np.ndarray, shape (n_particles, d)`` drawn
          from the prior.  The number of particles must match
          ``tmcmc_kwargs["n_particles"]``.

    tmcmc_kwargs : dict, optional
        Keyword arguments forwarded to :class:`~mcmckit.samplers.tmcmc.TMCMC`.
        Defaults: ``n_particles=500, n_mcmc_steps=3``.

    Examples
    --------
    Compare a 1-DOF and a 2-DOF structural model::

        mc_comp = mc.ModelComparison(
            models=[
                ("1-DOF", problem_1dof, prior_1dof),
                ("2-DOF", problem_2dof, prior_2dof),
            ],
        )
        mc_comp.run()
        mc_comp.summary()
    """

    def __init__(self, models: list, tmcmc_kwargs: dict | None = None):
        self._models = models
        self._tmcmc_kwargs = {"n_particles": 500, "n_mcmc_steps": 3}
        if tmcmc_kwargs:
            self._tmcmc_kwargs.update(tmcmc_kwargs)
        self._results: list[dict] = []

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> "ModelComparison":
        """Run TMCMC on all models and collect log-evidences.

        Returns
        -------
        self  (for chaining)
        """
        from ..samplers.tmcmc import TMCMC

        self._results = []
        for name, problem, prior_samples in self._models:
            sampler = TMCMC(**self._tmcmc_kwargs)
            result = sampler.run(problem, prior_samples)
            self._results.append({
                "name": name,
                "log_evidence": result.log_evidence,
                "result": result,
                "sampler": sampler,
            })

        # Sort best → worst (highest log-evidence first)
        self._results.sort(key=lambda r: r["log_evidence"], reverse=True)
        best_log_ev = self._results[0]["log_evidence"]

        for r in self._results:
            bf_info = bayes_factor(r["log_evidence"], best_log_ev)
            r["log_bf_vs_best"] = bf_info["log_bf"]
            r["log10_bf_vs_best"] = bf_info["log10_bf"]
            r["evidence_vs_best"] = bf_info["evidence"]

        return self

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, print_table: bool = True) -> list[dict]:
        """Print and return the model comparison table.

        Parameters
        ----------
        print_table : bool
            Whether to print the table to stdout.  Default True.

        Returns
        -------
        list of dict, one per model (sorted best → worst), with keys:
        ``name``, ``log_evidence``, ``log_bf_vs_best``, ``log10_bf_vs_best``,
        ``evidence_vs_best``.
        """
        if not self._results:
            raise RuntimeError("Call run() before summary().")

        if print_table:
            header = (f"{'Model':<20}  {'log p(y|M)':>12}  "
                      f"{'log10 BF':>10}  {'Evidence vs best':<25}")
            print(header)
            print("-" * len(header))
            for r in self._results:
                tag = " <-- best" if r["log10_bf_vs_best"] == 0.0 else ""
                print(
                    f"{r['name']:<20}  {r['log_evidence']:>12.3f}  "
                    f"{r['log10_bf_vs_best']:>10.2f}  "
                    f"{r['evidence_vs_best']:<25}{tag}"
                )

        return [
            {
                "name": r["name"],
                "log_evidence": r["log_evidence"],
                "log_bf_vs_best": r["log_bf_vs_best"],
                "log10_bf_vs_best": r["log10_bf_vs_best"],
                "evidence_vs_best": r["evidence_vs_best"],
            }
            for r in self._results
        ]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def best_model(self) -> str:
        """Name of the model with the highest log-evidence."""
        if not self._results:
            raise RuntimeError("Call run() before best_model().")
        return self._results[0]["name"]

    def get_result(self, name: str):
        """Return the :class:`~mcmckit.core.result.Result` for a named model."""
        if not self._results:
            raise RuntimeError("Call run() before get_result().")
        for r in self._results:
            if r["name"] == name:
                return r["result"]
        raise KeyError(f"No model named {name!r}.")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, title=None):
        """Bar chart of log-evidence (and log10 Bayes factors).

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if not self._results:
            raise RuntimeError("Call run() before plot().")

        names = [r["name"] for r in self._results]
        log_evs = np.array([r["log_evidence"] for r in self._results])
        log10_bfs = np.array([r["log10_bf_vs_best"] for r in self._results])

        x = np.arange(len(names))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        # Left: log-evidence
        axes[0].barh(x, log_evs, color="steelblue", alpha=0.8)
        axes[0].set_yticks(x)
        axes[0].set_yticklabels(names)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("log p(y | M)")
        axes[0].set_title("Log-evidence")

        # Right: log10 Bayes factor vs best
        colors = ["gold" if v == 0 else "steelblue" for v in log10_bfs]
        axes[1].barh(x, log10_bfs, color=colors, alpha=0.8)
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(names)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("log₁₀ BF vs best model")
        axes[1].set_title("Bayes factor (vs best)")

        # Reference lines for Jeffreys scale
        for val, lbl in [(-0.5, "Substantial"), (-1.0, "Strong"), (-2.0, "Decisive")]:
            axes[1].axvline(val, color="crimson", ls="--", lw=0.8, alpha=0.7)
            axes[1].text(val, len(names) - 0.5, lbl, color="crimson",
                         fontsize=6, ha="right", va="bottom", rotation=90)

        if title is not None:
            fig.suptitle(title)

        return fig

    def __repr__(self):
        n = len(self._models)
        ran = "run" if self._results else "not run"
        return f"ModelComparison(n_models={n}, status={ran!r})"
