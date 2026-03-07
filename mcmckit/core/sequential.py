from __future__ import annotations
from typing import Literal
import numpy as np


class PosteriorPrior:
    """Represent a previous posterior as a prior for sequential Bayesian updating.

    Wraps a set of posterior samples and exposes a callable ``log_prior``
    interface compatible with all mcmckit samplers.  Also provides a
    ``sample(n)`` method for use with ``TMCMC`` (which requires explicit prior
    samples).

    Two density estimation methods are supported:

    **Gaussian** (``method='gaussian'``)
        Fits a multivariate normal :math:`\\mathcal{N}(\\mu, \\Sigma)` to the
        samples.  Fast and exact in any dimension.  Works well when the
        posterior is approximately unimodal and elliptical — common in
        structural model updating.

        .. math::
            \\ln p(\\theta) = -\\tfrac{1}{2}(\\theta-\\mu)^T\\Sigma^{-1}(\\theta-\\mu)
                             - \\tfrac{d}{2}\\ln(2\\pi) - \\tfrac{1}{2}\\ln|\\Sigma|

    **KDE** (``method='kde'``)
        Fits a non-parametric kernel density estimate using
        ``scipy.stats.gaussian_kde`` with Scott's bandwidth rule.  More
        general: handles multimodal, skewed, or otherwise non-Gaussian
        posteriors.  Evaluation cost scales with the number of samples and
        is expensive above ~8 parameters.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, n_params)
        Posterior samples (already discarded / thinned as desired).
    method : {'gaussian', 'kde'}
        Density estimation method.  No default is provided — you must
        choose explicitly.
    regularise : float
        Small diagonal jitter added to the sample covariance before
        inversion (Gaussian mode only).  Prevents singular covariance for
        near-deterministic parameters.  Default 1e-8.

    Examples
    --------
    Gaussian approximation (any dimension)::

        prior2 = mc.PosteriorPrior(result1.discard(1000).samples, method='gaussian')

        problem2 = mc.Problem(prior=prior2, likelihood=ll_new)
        result2  = mc.DRAM(n_samples=20_000, initial_cov=prior2.cov).run(
            problem2, x0=prior2.mean
        )

    KDE (recommended for ≤ 6–8 parameters)::

        prior2 = mc.PosteriorPrior(result1.discard(1000).samples, method='kde')

    Use with TMCMC (requires samples from the prior)::

        result2 = mc.TMCMC(n_particles=1000).run(
            problem2, prior_samples=prior2.sample(1000)
        )

    Convenience shorthand via ``Result``::

        prior2 = result1.as_prior(method='gaussian', discard=1000)
    """

    def __init__(
        self,
        samples: np.ndarray,
        method: Literal["gaussian", "kde"],
        regularise: float = 1e-8,
    ):
        if method not in ("gaussian", "kde"):
            raise ValueError("method must be 'gaussian' or 'kde'.")

        self._samples = np.asarray(samples, dtype=float)
        if self._samples.ndim == 1:
            self._samples = self._samples[:, np.newaxis]
        self._n_samples, self._n_params = self._samples.shape
        self._method = method

        if method == "gaussian":
            self._mean = np.mean(self._samples, axis=0)
            cov = np.cov(self._samples.T)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            cov += regularise * np.eye(self._n_params)
            self._cov = cov
            # Precompute Cholesky and log-det for fast evaluation
            try:
                self._L = np.linalg.cholesky(self._cov)
            except np.linalg.LinAlgError:
                # fallback: add more jitter
                self._L = np.linalg.cholesky(
                    self._cov + 1e-6 * np.eye(self._n_params)
                )
            self._log_norm = (
                -0.5 * self._n_params * np.log(2.0 * np.pi)
                - np.sum(np.log(np.diag(self._L)))
            )

        elif method == "kde":
            try:
                from scipy.stats import gaussian_kde
            except ImportError as e:
                raise ImportError(
                    "scipy is required for KDE. Install it with: pip install scipy"
                ) from e
            self._kde = gaussian_kde(self._samples.T)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Density estimation method ('gaussian' or 'kde')."""
        return self._method

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self._n_params

    @property
    def mean(self) -> np.ndarray:
        """Sample mean, shape (n_params,)."""
        return self._mean if self._method == "gaussian" else np.mean(self._samples, axis=0)

    @property
    def cov(self) -> np.ndarray:
        """Sample covariance matrix, shape (n_params, n_params).

        For the Gaussian method this is the regularised covariance used
        internally.  Useful as ``initial_cov`` for the next sampler.
        """
        if self._method == "gaussian":
            return self._cov.copy()
        return np.cov(self._samples.T)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def __call__(self, theta) -> float:
        """Evaluate the log-prior at theta.

        Parameters
        ----------
        theta : array-like, shape (n_params,)

        Returns
        -------
        float
        """
        theta = np.asarray(theta, dtype=float)

        if self._method == "gaussian":
            diff = theta - self._mean
            z = np.linalg.solve(self._L, diff)
            return float(self._log_norm - 0.5 * np.dot(z, z))

        # KDE
        return float(self._kde.logpdf(theta.reshape(-1, 1))[0])

    # -------------------------------------------------------------------------
    # Sampling (for TMCMC prior_samples)
    # -------------------------------------------------------------------------

    def sample(self, n: int, rng=None) -> np.ndarray:
        """Draw n samples from the approximate prior.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        rng : np.random.Generator or None
            Optional random generator for reproducibility.

        Returns
        -------
        np.ndarray, shape (n, n_params)
        """
        if self._method == "gaussian":
            if rng is not None:
                return rng.multivariate_normal(self._mean, self._cov, size=n)
            return np.random.multivariate_normal(self._mean, self._cov, size=n)

        # KDE: resample from the kernel density estimate
        kde_samples = self._kde.resample(n, seed=rng)
        return kde_samples.T  # (n, n_params)

    # -------------------------------------------------------------------------
    # Repr
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PosteriorPrior(method='{self._method}', "
            f"n_params={self._n_params}, "
            f"n_samples={self._n_samples})"
        )
