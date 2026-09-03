import numpy as np

from ..core.parallel import WorkerPool, resolve_n_workers
from ..core.result import Result


class TMCMC:
    """Transitional Markov Chain Monte Carlo (TMCMC) sampler.

    Samples from the posterior by bridging from the prior to the posterior
    through a sequence of tempered intermediate distributions:

        p_j(θ) ∝ p(y|θ)^{β_j} p(θ),   0 = β_0 < β_1 < ... < β_J = 1

    At each stage:
    1. Compute importance weights w_j ∝ p(y|θ)^{Δβ} for the current particles
    2. Estimate the log-evidence contribution from this stage
    3. Resample particles according to the weights
    4. Rejuvenate particles with a few MH steps using the weighted covariance

    The log-evidence (log marginal likelihood) is accumulated across stages
    and returned in ``result.log_evidence``.

    Parameters
    ----------
    n_particles : int
        Number of particles (posterior samples). Typical: 500–2000.
    n_mcmc_steps : int
        Number of MH rejuvenation steps per particle per stage. Default 3.
    target_ess_ratio : float
        Target effective sample size as a fraction of n_particles when
        choosing the next β. Default 0.5 (50% ESS).
    n_workers : int, optional
        Number of parallel workers for likelihood evaluation. Default 1
        (sequential). Set > 1 to parallelize expensive forward models.
    cov_scale : float
        Scaling of the weighted covariance for the MH proposal.
        Default 2.38²/d (theoretically optimal).

    Notes
    -----
    TMCMC does not implement ``step()`` in the single-sample sense — it
    operates in stages. Use ``run_stage()`` for stage-by-stage execution.

    References
    ----------
    Ching, J., & Chen, Y. C. (2007). Transitional Markov chain Monte Carlo
    method for Bayesian model updating, model class selection, and model
    averaging. Journal of Engineering Mechanics, 133(7), 816-832.
    """

    def __init__(
        self,
        n_particles,
        n_mcmc_steps=3,
        target_ess_ratio=0.5,
        n_workers=1,
        cov_scale=None,
        backend="auto",
    ):
        self.n_particles = n_particles
        self.n_mcmc_steps = n_mcmc_steps
        self.target_ess_ratio = target_ess_ratio
        self.n_workers = resolve_n_workers(n_workers)
        self.backend = backend
        self._cov_scale = cov_scale
        self._initialized = False
        self._pool = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self, problem):
        """Sample initial particles from the prior and evaluate likelihoods.

        Parameters
        ----------
        problem : Problem
            The problem must expose ``problem.prior`` as a callable that can
            also be used to *sample* from the prior, OR the user can call
            ``initialize_with_samples(problem, samples)`` to provide
            prior samples directly.
        """
        raise NotImplementedError(
            "TMCMC cannot sample the prior automatically — call "
            "initialize_with_samples(problem, prior_samples) instead, "
            "where prior_samples is an (n_particles, d) array drawn from p(θ)."
        )

    def initialize_with_samples(self, problem, prior_samples):
        """Initialize TMCMC with user-provided prior samples.

        Parameters
        ----------
        problem : Problem
        prior_samples : array-like, shape (n_particles, d)
            Samples drawn from the prior p(θ).
        """
        prior_samples = np.asarray(prior_samples, dtype=float)
        if prior_samples.shape[0] != self.n_particles:
            raise ValueError(
                f"prior_samples has {prior_samples.shape[0]} rows but "
                f"n_particles={self.n_particles}"
            )
        self._problem = problem
        self._particles = prior_samples.copy()
        self._d = prior_samples.shape[1]
        self._log_likelihoods = self._eval_log_likelihoods(self._particles)
        self._beta = 0.0
        self._log_evidence = 0.0
        self._stage = 0
        self._history = [(0.0, prior_samples.copy())]  # (beta, particles)
        self._initialized = True

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    def run_stage(self):
        """Advance one tempering stage. Returns the new beta value."""
        if not self._initialized:
            raise RuntimeError("Call initialize_with_samples() before run_stage().")
        if self._beta >= 1.0:
            return self._beta  # already done

        # Find next beta via bisection
        delta_beta = self._find_delta_beta(self._log_likelihoods, self._beta)
        new_beta = min(self._beta + delta_beta, 1.0)
        actual_delta = new_beta - self._beta

        # Incremental importance weights
        log_w = actual_delta * self._log_likelihoods
        log_w -= np.max(log_w)               # numerical stability
        w = np.exp(log_w)
        w_norm = w / w.sum()

        # Log-evidence contribution: log E[p(y|θ)^Δβ]
        # = log(mean(exp(Δβ * ll))) computed in a numerically stable way
        max_ll = np.max(self._log_likelihoods)
        self._log_evidence += (np.log(np.mean(np.exp(actual_delta * self._log_likelihoods
                                                      - actual_delta * max_ll)))
                               + actual_delta * max_ll)

        # Weighted covariance for MH rejuvenation
        cov_scale = self._cov_scale if self._cov_scale is not None else 2.38**2 / self._d
        weighted_mean = np.average(self._particles, weights=w_norm, axis=0)
        diff = self._particles - weighted_mean
        weighted_cov = cov_scale * (diff.T * w_norm) @ diff
        weighted_cov += 1e-6 * np.eye(self._d)  # regularize

        # Systematic resampling
        indices = _systematic_resample(w_norm, self.n_particles)
        self._particles = self._particles[indices].copy()
        self._log_likelihoods = self._log_likelihoods[indices].copy()

        # MH rejuvenation
        self._particles, self._log_likelihoods = self._rejuvenate(
            self._particles, self._log_likelihoods, weighted_cov, new_beta
        )

        self._beta = new_beta
        self._stage += 1
        self._history.append((self._beta, self._particles.copy()))
        return self._beta

    def run(self, problem, prior_samples):
        """Run all stages from prior samples to posterior.

        Parameters
        ----------
        problem : Problem
        prior_samples : array-like, shape (n_particles, d)
            Samples drawn from the prior.

        Returns
        -------
        Result
            Posterior samples with ``result.log_evidence`` set.
        """
        pool = WorkerPool(
            n_workers=self.n_workers,
            backend=self.backend,
            func=problem.log_likelihood,
        )
        # One pool for the whole run. Rebuilding it per stage costs far more
        # than it saves, especially on Windows where each worker re-imports the
        # calling module.
        with pool:
            self._pool = pool
            try:
                self.initialize_with_samples(problem, prior_samples)
                while self._beta < 1.0:
                    self.run_stage()
            finally:
                self._pool = None
        return self.get_result()

    def get_result(self):
        """Return Result from current particles."""
        if not self._initialized:
            raise RuntimeError("No samples yet.")
        log_posteriors = (self._log_likelihoods
                          + np.array([self._problem.log_prior(p) for p in self._particles]))
        return Result(
            samples=self._particles.copy(),
            log_posteriors=log_posteriors,
            param_names=self._problem.param_names,
            acceptance_rate=None,
            log_evidence=self._log_evidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_log_likelihoods(self, particles):
        """Evaluate the log-likelihood for every particle.

        Uses the worker pool opened by :meth:`run` when there is one. Outside
        that context - e.g. when driving stages by hand with
        :meth:`run_stage` - it falls back to serial evaluation, so results are
        identical either way.
        """
        if self._pool is not None:
            results = self._pool.map(self._problem.log_likelihood, particles)
        else:
            results = [self._problem.log_likelihood(p) for p in particles]
        return np.array(results)

    def _find_delta_beta(self, log_likelihoods, beta_current):
        """Bisection to find Δβ such that ESS/N ≈ target_ess_ratio."""
        target_ess = self.target_ess_ratio * self.n_particles

        def ess(delta):
            log_w = delta * log_likelihoods
            log_w -= np.max(log_w)
            w = np.exp(log_w)
            w /= w.sum()
            return 1.0 / np.sum(w**2)

        # Check if Δβ = 1 - β_current already meets target
        max_delta = 1.0 - beta_current
        if ess(max_delta) >= target_ess:
            return max_delta

        # Bisect
        lo, hi = 0.0, max_delta
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if ess(mid) < target_ess:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-6:
                break
        return lo

    def _rejuvenate(self, particles, log_likelihoods, cov, beta):
        """MH rejuvenation: run n_mcmc_steps MH steps per particle."""
        new_particles = particles.copy()
        new_ll = log_likelihoods.copy()

        for _ in range(self.n_mcmc_steps):
            proposals = np.array([
                np.random.multivariate_normal(p, cov) for p in new_particles
            ])
            prop_ll = self._eval_log_likelihoods(proposals)
            prop_lprior = np.array([self._problem.log_prior(p) for p in proposals])
            curr_lprior = np.array([self._problem.log_prior(p) for p in new_particles])

            log_alpha = (beta * prop_ll + prop_lprior) - (beta * new_ll + curr_lprior)
            u = np.log(np.random.rand(self.n_particles))
            accept = u < log_alpha

            new_particles[accept] = proposals[accept]
            new_ll[accept] = prop_ll[accept]

        return new_particles, new_ll

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_stages(self, max_stages=6, kde_grid=80, levels=4, title=None):
        """Corner-style plot showing particle evolution across tempering stages.

        Each stage is drawn as a KDE (diagonal: 1-D curve; lower triangle:
        2-D contours), colored light→dark as β goes from 0 to 1.
        Upper triangle is hidden.

        Parameters
        ----------
        max_stages : int, optional
            Maximum number of stages to overlay (evenly subsampled, always
            including β=0 and β=1).  Default 6 keeps the plot readable.
        kde_grid : int
            Grid resolution for KDE evaluation.
        levels : int
            Number of contour levels for 2-D KDE panels.
        title : str, optional
            Figure suptitle.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        if not self._history:
            raise RuntimeError("No history — run the sampler first.")

        history = self._history
        if len(history) > max_stages:
            idx = np.round(np.linspace(0, len(history) - 1, max_stages)).astype(int)
            history = [history[i] for i in idx]

        n_stages = len(history)
        d = history[0][1].shape[1]
        names = (self._problem.param_names if self._problem.param_names
                 else [f"theta[{i}]" for i in range(d)])

        # Colormap: light (prior) → dark (posterior)
        cmap = plt.cm.plasma_r
        colors = [cmap(0.1 + 0.85 * k / max(n_stages - 1, 1)) for k in range(n_stages)]

        fig, axes = plt.subplots(d, d, figsize=(2.8 * d, 2.8 * d), constrained_layout=True)
        if d == 1:
            axes = np.array([[axes]])

        for row in range(d):
            for col in range(d):
                ax = axes[row, col]

                if row == col:
                    # Diagonal: overlaid 1-D KDE curves
                    for k, (beta, pts) in enumerate(history):
                        x = pts[:, col]
                        try:
                            kde = gaussian_kde(x)
                            xi = np.linspace(x.min(), x.max(), kde_grid)
                            lw = 1.0 + 0.8 * k / max(n_stages - 1, 1)
                            ax.plot(xi, kde(xi), color=colors[k], lw=lw,
                                    label=f"β={beta:.3f}")
                        except Exception:
                            pass

                elif row > col:
                    # Lower triangle: 2-D KDE contours per stage
                    for k, (beta, pts) in enumerate(history):
                        x, y = pts[:, col], pts[:, row]
                        try:
                            kde = gaussian_kde(np.vstack([x, y]))
                            xi = np.linspace(x.min(), x.max(), kde_grid)
                            yi = np.linspace(y.min(), y.max(), kde_grid)
                            xx, yy = np.meshgrid(xi, yi)
                            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                            lw = 0.8 + 0.6 * k / max(n_stages - 1, 1)
                            ax.contour(xx, yy, zz, levels=levels,
                                       colors=[colors[k]], linewidths=lw, alpha=0.9)
                        except Exception:
                            pass

                else:
                    ax.set_visible(False)
                    continue

                if row == d - 1:
                    ax.set_xlabel(names[col], fontsize=8)
                else:
                    ax.set_xticklabels([])
                if col == 0 and row != 0:
                    ax.set_ylabel(names[row], fontsize=8)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=7)

        axes[0, 0].legend(fontsize=6, loc="upper right", framealpha=0.7,
                          title="β", title_fontsize=6)

        if title is not None:
            fig.suptitle(title)
        return fig

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def beta(self):
        """Current tempering parameter β."""
        return self._beta

    @property
    def stage(self):
        """Number of completed tempering stages."""
        return self._stage

    @property
    def log_evidence(self):
        """Accumulated log-evidence estimate."""
        return self._log_evidence


# ------------------------------------------------------------------
# Resampling utility
# ------------------------------------------------------------------

def _systematic_resample(weights, n):
    """Systematic resampling — low variance, O(n)."""
    positions = (np.arange(n) + np.random.uniform()) / n
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return np.clip(indices, 0, len(weights) - 1)
