import numpy as np


def _kde2d(x, y, grid_size=100):
    """Return (xx, yy, zz) for a 2D KDE evaluated on a meshgrid."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(np.vstack([x, y]))
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(xi, yi)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def _kde1d(x, grid_size=200):
    """Return (xi, zi) for a 1D KDE."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x)
    xi = np.linspace(x.min(), x.max(), grid_size)
    return xi, kde(xi)


def _subsample_indices(n_total, n_eval):
    """Return sorted indices for subsampling posterior samples.

    Currently: uniform random subsampling without replacement.

    EXTENSION POINT — swap this function to use a smarter strategy for
    expensive forward models, e.g.:
      - LHS-based stratification over the posterior CDF
      - Systematic resampling (divide chain into n_eval equal segments)
      - Importance resampling (weight by posterior density)
    All return the same contract: a sorted int array of length <= n_total.
    """
    if n_eval is None or n_eval >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.choice(n_total, size=n_eval, replace=False))


class Result:
    """Container for posterior samples produced by a sampler.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, n_params)
    log_posteriors : np.ndarray, shape (n_samples,)
    param_names : list of str, optional
    acceptance_rate : float, optional
    """

    def __init__(self, samples, log_posteriors, param_names=None, acceptance_rate=None,
                 log_evidence=None):
        self.samples = np.asarray(samples)
        self.log_posteriors = np.asarray(log_posteriors)
        self.param_names = param_names
        self.acceptance_rate = acceptance_rate
        self.log_evidence = log_evidence  # set by TMCMC; None for MCMC samplers

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def mean(self):
        return np.mean(self.samples, axis=0)

    def std(self):
        return np.std(self.samples, axis=0)

    def cov(self):
        return np.cov(self.samples.T)

    def quantile(self, q):
        return np.quantile(self.samples, q, axis=0)

    def select(self, indices):
        """Return a new Result containing only a subset of parameters.

        Parameters
        ----------
        indices : array-like of int or str
            Parameter indices (int) or names (str) to keep.

        Returns
        -------
        Result

        Examples
        --------
        ::

            r_sub = result.select([0, 2, 6, 9])          # by index
            r_sub = result.select(["k1", "k3", "k7"])    # by name
        """
        idx = []
        for i in indices:
            if isinstance(i, str):
                if self.param_names is None:
                    raise ValueError("param_names not set; use integer indices.")
                idx.append(self.param_names.index(i))
            else:
                idx.append(int(i))
        names = [self.param_names[i] for i in idx] if self.param_names else None
        return Result(self.samples[:, idx], self.log_posteriors,
                      param_names=names, acceptance_rate=self.acceptance_rate)

    def discard(self, n):
        """Return a new Result with the first n samples removed (burn-in).

        Parameters
        ----------
        n : int
            Number of initial samples to discard.
        """
        if n >= len(self.samples):
            raise ValueError(f"Cannot discard {n} samples from a chain of length {len(self.samples)}.")
        return Result(
            samples=self.samples[n:],
            log_posteriors=self.log_posteriors[n:],
            param_names=self.param_names,
            acceptance_rate=self.acceptance_rate,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def ess(self):
        """Effective sample size per parameter.

        Returns
        -------
        np.ndarray, shape (n_params,)

        See Also
        --------
        mcmckit.core.diagnostics.ess
        """
        from .diagnostics import ess as _ess
        return _ess(self.samples)

    def autocorr(self, max_lag=100):
        """Normalised autocorrelation function for each parameter.

        Parameters
        ----------
        max_lag : int
            Maximum lag to compute.

        Returns
        -------
        np.ndarray, shape (max_lag + 1, n_params)

        See Also
        --------
        mcmckit.core.diagnostics.autocorr
        """
        from .diagnostics import autocorr as _autocorr
        return _autocorr(self.samples, max_lag=max_lag)

    def plot_autocorr(self, max_lag=100, title=None):
        """Plot the autocorrelation function for each parameter.

        Parameters
        ----------
        max_lag : int
            Maximum lag to display.
        title : str, optional

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        acf = self.autocorr(max_lag=max_lag)
        lags = np.arange(acf.shape[0])
        n_params = acf.shape[1]
        names = self.param_names or [f"theta[{i}]" for i in range(n_params)]

        ncols = min(n_params, 3)
        nrows = (n_params + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows),
                                 squeeze=False, constrained_layout=True)
        axes_flat = axes.flatten()

        for i in range(n_params):
            ax = axes_flat[i]
            ax.bar(lags, acf[:, i], width=1.0, color="steelblue", alpha=0.7)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xlabel("lag")
            ax.set_ylabel("ACF")
            ax.set_title(names[i])

        for j in range(n_params, len(axes_flat)):
            axes_flat[j].set_visible(False)

        if title is not None:
            fig.suptitle(title)
        return fig

    # ------------------------------------------------------------------
    # Visualisation (requires matplotlib)
    # ------------------------------------------------------------------

    def plot_trace(self, title=None, **kwargs):
        import matplotlib.pyplot as plt

        n_params = self.samples.shape[1]
        names = self.param_names or [f"theta[{i}]" for i in range(n_params)]

        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params), squeeze=False,
                                 constrained_layout=True)
        for i, ax in enumerate(axes[:, 0]):
            ax.plot(self.samples[:, i], lw=0.7, **kwargs)
            ax.set_ylabel(names[i])
        axes[-1, 0].set_xlabel("iteration")
        if title is not None:
            fig.suptitle(title)
        return fig

    def plot_marginals(self, bins=40, title=None, **kwargs):
        import matplotlib.pyplot as plt

        n_params = self.samples.shape[1]
        names = self.param_names or [f"theta[{i}]" for i in range(n_params)]

        ncols = min(n_params, 3)
        nrows = (n_params + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False,
                                 constrained_layout=True)
        axes_flat = axes.flatten()

        for i in range(n_params):
            axes_flat[i].hist(self.samples[:, i], bins=bins, density=True, **kwargs)
            axes_flat[i].set_xlabel(names[i])
            axes_flat[i].set_ylabel("density")

        for j in range(n_params, len(axes_flat)):
            axes_flat[j].set_visible(False)

        if title is not None:
            fig.suptitle(title)
        return fig

    def plot_corner(
        self,
        style="corner",
        bins=30,
        kde_grid=80,
        levels=6,
        true_values=None,
        title=None,
        scatter_kwargs=None,
        hist_kwargs=None,
        kde_kwargs=None,
    ):
        """Corner / pair plot of posterior samples.

        Parameters
        ----------
        style : str
            Layout style:

            ``"corner"`` *(default)*
                Diagonal = histogram + KDE. Lower triangle = 2D KDE contours.
                Upper triangle = empty.

            ``"scatter"``
                Diagonal = histogram. Lower triangle = scatter plot.
                Upper triangle = empty.

            ``"full"``
                Diagonal = histogram + KDE. Lower triangle = scatter.
                Upper triangle = 2D KDE contours.

            ``"kde"``
                Diagonal = KDE only. All off-diagonal = 2D KDE contours.

        bins : int
            Number of histogram bins for diagonal panels.
        kde_grid : int
            Grid resolution for 2D KDE evaluation.
        levels : int
            Number of contour levels for 2D KDE panels.
        true_values : array-like, optional
            True / reference parameter values. Shown as a vertical line on
            diagonal panels and a cross on off-diagonal panels.
        scatter_kwargs : dict, optional
            Passed to ``ax.scatter`` for scatter panels.
        hist_kwargs : dict, optional
            Passed to ``ax.hist`` for diagonal panels.
        kde_kwargs : dict, optional
            Passed to ``ax.contourf`` for KDE contour panels.
        """
        import matplotlib.pyplot as plt

        _scatter_kw = dict(s=1, alpha=0.3, color="steelblue", rasterized=True)
        _scatter_kw.update(scatter_kwargs or {})
        _hist_kw = dict(bins=bins, density=True, color="steelblue", alpha=0.6)
        _hist_kw.update(hist_kwargs or {})
        _kde_kw = dict(levels=levels, cmap="Blues")
        _kde_kw.update(kde_kwargs or {})

        valid = {"corner", "scatter", "full", "kde"}
        if style not in valid:
            raise ValueError(f"style must be one of {valid}, got {style!r}")

        true_values = np.asarray(true_values) if true_values is not None else None

        n = self.samples.shape[1]
        names = self.param_names or [f"theta[{i}]" for i in range(n)]
        means = self.mean()
        stds = self.std()

        fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n), constrained_layout=True)
        if n == 1:
            axes = np.array([[axes]])

        for row in range(n):
            for col in range(n):
                ax = axes[row, col]
                xi = self.samples[:, col]
                yi = self.samples[:, row]

                if row == col:
                    # --- diagonal ---
                    if style == "kde":
                        xg, zg = _kde1d(xi)
                        ax.plot(xg, zg, color="steelblue", lw=1.5)
                        ax.fill_between(xg, zg, alpha=0.25, color="steelblue")
                    else:
                        ax.hist(xi, **_hist_kw)
                        if style in ("corner", "full"):
                            xg, zg = _kde1d(xi)
                            ax.plot(xg, zg, color="navy", lw=1.2)

                    # mean ± std as title
                    ax.set_title(f"{means[col]:.3g} ± {stds[col]:.3g}", fontsize=7, pad=2)

                    # true value: vertical line
                    if true_values is not None:
                        ax.axvline(true_values[col], color="crimson", lw=1.2, ls="--", zorder=5)

                elif row > col:
                    # --- lower triangle ---
                    if style in ("scatter", "full"):
                        ax.scatter(xi, yi, **_scatter_kw)
                    else:  # "corner" or "kde"
                        xx, yy, zz = _kde2d(xi, yi, grid_size=kde_grid)
                        ax.contourf(xx, yy, zz, **_kde_kw)
                        ax.contour(xx, yy, zz, levels=levels, colors="navy", linewidths=0.5, alpha=0.6)

                    # true value: cross
                    if true_values is not None:
                        ax.plot(true_values[col], true_values[row], marker="+",
                                color="crimson", ms=8, mew=1.5, zorder=5, ls="none")

                else:
                    # --- upper triangle ---
                    if style in ("corner", "scatter"):
                        ax.set_visible(False)
                        continue
                    else:  # "full" or "kde"
                        xx, yy, zz = _kde2d(xi, yi, grid_size=kde_grid)
                        ax.contourf(xx, yy, zz, **_kde_kw)
                        ax.contour(xx, yy, zz, levels=levels, colors="navy", linewidths=0.5, alpha=0.6)

                    # true value: cross
                    if true_values is not None:
                        ax.plot(true_values[col], true_values[row], marker="+",
                                color="crimson", ms=8, mew=1.5, zorder=5, ls="none")

                # axis labels on edges only
                if row == n - 1:
                    ax.set_xlabel(names[col], fontsize=8)
                else:
                    ax.set_xticklabels([])

                if col == 0 and row != 0:
                    ax.set_ylabel(names[row], fontsize=8)
                else:
                    ax.set_yticklabels([])

                ax.tick_params(labelsize=7)

        if title is not None:
            fig.suptitle(title)
        return fig

    # ------------------------------------------------------------------
    # Posterior predictive
    # ------------------------------------------------------------------

    def posterior_predictive(self, forward_model, n_eval=None):
        """Evaluate the forward model at posterior samples.

        Parameters
        ----------
        forward_model : callable
            ``f(theta) -> array-like, shape (n_obs,)``.
        n_eval : int, optional
            Number of posterior samples to evaluate.  ``None`` (default)
            evaluates all samples.  For expensive forward models (e.g. FEM),
            use a smaller ``n_eval`` — see ``_subsample_indices`` for the
            extension point to plug in smarter sampling strategies.

        Returns
        -------
        PosteriorPredictive
        """
        idx = _subsample_indices(len(self.samples), n_eval)
        theta_sub = self.samples[idx]
        preds = np.array([np.asarray(forward_model(t), dtype=float).ravel()
                          for t in theta_sub])
        return PosteriorPredictive(preds, theta_sub,
                                   param_names=self.param_names)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def as_prior(self, method: str, discard: int = 0):
        """Convert this result into a prior for the next sequential update step.

        Parameters
        ----------
        method : {'gaussian', 'kde'}
            Density estimation method.  ``'gaussian'`` fits a multivariate
            normal and is fast in any dimension.  ``'kde'`` fits a
            non-parametric kernel density estimate (recommended for ≤ 8
            parameters when the posterior is non-Gaussian or multimodal).
        discard : int
            Number of initial samples to discard as burn-in before fitting.

        Returns
        -------
        PosteriorPrior
            A callable that evaluates ``log p(theta)`` and supports
            ``sample(n)`` for use with TMCMC.

        Examples
        --------
        ::

            prior2 = result1.as_prior(method='gaussian', discard=1000)
            problem2 = mc.Problem(prior=prior2, likelihood=ll_new)
            result2 = mc.DRAM(n_samples=20_000, initial_cov=prior2.cov).run(
                problem2, x0=prior2.mean
            )
        """
        from .sequential import PosteriorPrior
        samples = self.discard(discard).samples
        return PosteriorPrior(samples, method=method)

    def __repr__(self):
        n, d = self.samples.shape
        ar = f", acceptance_rate={self.acceptance_rate:.3f}" if self.acceptance_rate is not None else ""
        return f"Result(n_samples={n}, n_params={d}{ar})"


class PosteriorPredictive:
    """Container for posterior predictive samples from a forward model.

    Produced by :meth:`Result.posterior_predictive`.

    Parameters
    ----------
    predictions : np.ndarray, shape (n_eval, n_obs)
        Forward model outputs at each sampled theta.
    theta : np.ndarray, shape (n_eval, n_params)
        The posterior samples used to compute the predictions.
    param_names : list of str, optional
    """

    def __init__(self, predictions, theta, param_names=None):
        self.predictions = np.asarray(predictions)
        self.theta = np.asarray(theta)
        self.param_names = param_names

    def mean(self):
        """Pointwise mean of predictions, shape (n_obs,)."""
        return np.mean(self.predictions, axis=0)

    def std(self):
        """Pointwise std of predictions, shape (n_obs,)."""
        return np.std(self.predictions, axis=0)

    def quantile(self, q):
        """Pointwise quantile(s) of predictions.

        Parameters
        ----------
        q : float or array-like
            Quantile(s) in [0, 1].

        Returns
        -------
        np.ndarray, shape (n_obs,) or (len(q), n_obs)
        """
        return np.quantile(self.predictions, q, axis=0)

    def plot_bands(self, x=None, ci=(0.05, 0.95), y_obs=None,
                   obs_kwargs=None, band_kwargs=None,
                   title=None, xlabel=None, ylabel=None, ax=None):
        """Plot median + credible band of posterior predictive.

        Parameters
        ----------
        x : array-like, optional
            x-axis values (e.g. frequency vector, time vector).
            Defaults to integer indices 0, 1, …, n_obs-1.
        ci : tuple of two floats
            Lower and upper quantile levels for the credible band.
            Default (0.05, 0.95) gives a 90% band.
        y_obs : array-like, optional
            Observed data to overlay as scatter points.
        obs_kwargs : dict, optional
            Passed to ``ax.scatter`` for observed data.
        band_kwargs : dict, optional
            Passed to ``ax.fill_between`` for the credible band.
        title : str, optional
        xlabel : str, optional
        ylabel : str, optional
        ax : matplotlib Axes, optional
            If None, a new figure is created.

        Returns
        -------
        matplotlib Figure (or None if ax was provided)
        """
        import matplotlib.pyplot as plt

        n_obs = self.predictions.shape[1]
        x = np.arange(n_obs) if x is None else np.asarray(x)

        _band_kw = dict(alpha=0.25, color="steelblue", label=f"{int((ci[1]-ci[0])*100)}% CI")
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
        ax.plot(x, med, color="steelblue", lw=1.5, label="median")

        if y_obs is not None:
            ax.scatter(x, np.asarray(y_obs), **_obs_kw)

        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        ax.legend(fontsize=8)
        if title is not None and fig is not None:
            fig.suptitle(title)

        return fig

    def __repr__(self):
        n_eval, n_obs = self.predictions.shape
        return f"PosteriorPredictive(n_eval={n_eval}, n_obs={n_obs})"
