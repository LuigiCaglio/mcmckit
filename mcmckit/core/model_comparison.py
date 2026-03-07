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

    # ------------------------------------------------------------------
    # Model averaging
    # ------------------------------------------------------------------

    def weights(self, prior_weights=None) -> np.ndarray:
        """Posterior model probabilities (Bayesian Model Averaging weights).

        With equal prior model probabilities:

        .. math::

            w_k = \\frac{p(y \\mid M_k)}{\\sum_j p(y \\mid M_j)}

        Parameters
        ----------
        prior_weights : array-like of float, optional
            Prior model probabilities :math:`p(M_k)`, in the same order as
            ``summary()`` output (sorted best → worst by evidence).
            Defaults to uniform (1/K for each model).

        Returns
        -------
        np.ndarray, shape (n_models,)
            Posterior model weights, summing to 1.  Order matches
            ``summary()`` (best model first).
        """
        if not self._results:
            raise RuntimeError("Call run() before weights().")
        log_evs = np.array([r["log_evidence"] for r in self._results],
                           dtype=float)
        if prior_weights is not None:
            pw = np.asarray(prior_weights, dtype=float)
            if len(pw) != len(self._results):
                raise ValueError(
                    f"prior_weights length {len(pw)} != n_models {len(self._results)}"
                )
            log_evs = log_evs + np.log(pw / pw.sum())
        log_evs -= np.max(log_evs)   # numerical stability
        w = np.exp(log_evs)
        return w / w.sum()

    def predict(self, forward_models, n_eval: int = 500) -> "BMAResult":
        """Bayesian Model Averaged posterior predictive.

        Draws ``round(w_k * n_eval)`` samples from each model's posterior,
        evaluates the corresponding forward model, and returns a
        :class:`BMAResult` containing the mixture predictions.

        Parameters
        ----------
        forward_models : dict or list
            Forward model callables, one per model.

            * **dict** ``{name: callable}`` — keyed by the model name strings
              passed to the constructor.
            * **list** of callables — in the same order as ``summary()`` output
              (sorted best → worst by evidence).

        n_eval : int
            Total number of forward model evaluations to distribute across
            models proportionally to their weights.  Default 500.

        Returns
        -------
        BMAResult

        Examples
        --------
        ::

            bma = comp.predict(
                forward_models={
                    "M1: 1-DOF": lambda theta: [omega(theta[0]), omega(theta[0])],
                    "M2: 2-DOF": lambda theta: natural_frequencies(theta[0], theta[1]),
                },
                n_eval=1000,
            )
            bma.plot_bands(y_obs=y_obs)
        """
        if not self._results:
            raise RuntimeError("Call run() before predict().")

        w = self.weights()
        model_preds = []

        for i, r in enumerate(self._results):
            n_k = max(1, round(float(w[i]) * n_eval))
            if isinstance(forward_models, dict):
                fwd = forward_models[r["name"]]
            else:
                fwd = forward_models[i]
            pp = r["result"].posterior_predictive(fwd, n_eval=n_k)
            model_preds.append(pp.predictions)

        return BMAResult(
            model_predictions=model_preds,
            weights=w,
            model_names=[r["name"] for r in self._results],
        )

    def __repr__(self):
        n = len(self._models)
        ran = "run" if self._results else "not run"
        return f"ModelComparison(n_models={n}, status={ran!r})"


# ------------------------------------------------------------------
# BMAResult
# ------------------------------------------------------------------

class BMAResult:
    """Posterior predictive distribution from Bayesian Model Averaging.

    Produced by :meth:`ModelComparison.predict`.  Stores predictions from
    each model separately and exposes mixture statistics and plots.

    Parameters
    ----------
    model_predictions : list of np.ndarray, each shape (n_k, n_obs)
        Forward model outputs at posterior samples, one array per model.
    weights : np.ndarray, shape (n_models,)
        Posterior model probabilities.
    model_names : list of str

    Notes
    -----
    Statistics (``mean``, ``std``, ``quantile``) are computed on the
    concatenated mixture sample, where model k contributes
    ``round(w_k * n_eval)`` samples.  When one model is decisive
    (weight ≈ 1) the BMA prediction effectively collapses to that model.
    """

    def __init__(self, model_predictions: list, weights: np.ndarray,
                 model_names: list):
        self.model_predictions = [np.asarray(p, dtype=float)
                                   for p in model_predictions]
        self.weights = np.asarray(weights, dtype=float)
        self.model_names = list(model_names)
        # Concatenated mixture sample
        self.predictions = np.concatenate(self.model_predictions, axis=0)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def mean(self) -> np.ndarray:
        """Pointwise mean of the BMA predictive, shape (n_obs,)."""
        return np.mean(self.predictions, axis=0)

    def std(self) -> np.ndarray:
        """Pointwise std of the BMA predictive, shape (n_obs,)."""
        return np.std(self.predictions, axis=0)

    def quantile(self, q) -> np.ndarray:
        """Pointwise quantile(s) of the BMA predictive.

        Parameters
        ----------
        q : float or array-like

        Returns
        -------
        np.ndarray, shape (n_obs,) or (len(q), n_obs)
        """
        return np.quantile(self.predictions, q, axis=0)

    def decompose(self) -> dict:
        """Per-model contribution: weight, mean prediction, and std.

        Returns
        -------
        dict keyed by model name, each value a dict with keys
        ``weight``, ``mean``, ``std``.
        """
        return {
            name: {
                "weight": float(w),
                "mean": p.mean(axis=0),
                "std": p.std(axis=0),
            }
            for name, w, p in zip(self.model_names, self.weights,
                                   self.model_predictions)
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_bands(self, x=None, ci=(0.05, 0.95), y_obs=None,
                   obs_kwargs=None, band_kwargs=None,
                   title=None, xlabel=None, ylabel=None, ax=None):
        """Credible band of the BMA posterior predictive.

        Parameters
        ----------
        x : array-like, optional
        ci : tuple of two floats
        y_obs : array-like, optional
        obs_kwargs, band_kwargs : dict, optional
        title, xlabel, ylabel : str, optional
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib Figure (or None if ax was provided)
        """
        import matplotlib.pyplot as plt

        n_obs = self.predictions.shape[1]
        x = np.arange(n_obs) if x is None else np.asarray(x)

        _band_kw = dict(alpha=0.25, color="steelblue",
                        label=f"{int((ci[1] - ci[0]) * 100)}% CI (BMA)")
        _band_kw.update(band_kwargs or {})
        _obs_kw = dict(s=20, color="crimson", zorder=5, label="observed")
        _obs_kw.update(obs_kwargs or {})

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

        lo = self.quantile(ci[0])
        hi = self.quantile(ci[1])
        med = self.quantile(0.5)

        ax.fill_between(x, lo, hi, **_band_kw)
        ax.plot(x, med, color="steelblue", lw=1.5, label="BMA median")

        if y_obs is not None:
            ax.scatter(x, np.asarray(y_obs), **_obs_kw)

        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        ax.legend(fontsize=8)
        if title is not None and fig is not None:
            fig.suptitle(title)
        return fig

    def plot_decompose(self, x=None, title=None, xlabel=None, ylabel=None):
        """Per-model mean predictions with line width proportional to weight.

        Helps visualise how much each model contributes to the BMA.

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        n_obs = self.predictions.shape[1]
        x = np.arange(n_obs) if x is None else np.asarray(x)

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        cmap = plt.cm.tab10
        for i, (name, w, p) in enumerate(zip(self.model_names, self.weights,
                                              self.model_predictions)):
            lw = 0.5 + 4.0 * float(w)   # thicker line = higher weight
            ax.plot(x, p.mean(axis=0), lw=lw, color=cmap(i / 10),
                    label=f"{name}  (w={w:.3f})", alpha=0.85)

        bma_mean = self.mean()
        ax.plot(x, bma_mean, "k--", lw=1.5, label="BMA mean")

        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        ax.legend(fontsize=8)
        if title is not None:
            fig.suptitle(title)
        return fig

    def __repr__(self) -> str:
        K = len(self.model_names)
        n_obs = self.predictions.shape[1]
        w_str = ", ".join(f"{n}: {w:.3f}"
                          for n, w in zip(self.model_names, self.weights))
        return f"BMAResult(n_models={K}, n_obs={n_obs}, weights=[{w_str}])"
