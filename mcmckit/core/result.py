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
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        n, d = self.samples.shape
        ar = f", acceptance_rate={self.acceptance_rate:.3f}" if self.acceptance_rate is not None else ""
        return f"Result(n_samples={n}, n_params={d}{ar})"
