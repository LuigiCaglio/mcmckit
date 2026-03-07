import numpy as np


class GaussianNoiseLikelihood:
    """Log-likelihood for a forward model with Gaussian observation noise.

    Three modes, selected by the constructor arguments:

    **Fixed σ** (``noise_std`` is a float):
        σ is known.  ``theta`` contains only model parameters.

        log p(y|θ) = -0.5 * ||y - f(θ)||² / σ² - n·log(σ)

    **Estimated σ** (``noise_std=None``, ``marginalise_noise=False``):
        σ is a free parameter.  The *last element* of ``theta`` is
        ``log(σ)`` and is sampled jointly with the model parameters.

        log p(y|θ, log σ) = -0.5 * ||y - f(θ)||² / σ² - n·log(σ)

    **Marginalised σ** (``noise_std=None``, ``marginalise_noise=True``):
        σ² is given an Inverse-Gamma(α, β) prior and integrated out
        analytically.  ``theta`` contains *only* model parameters —
        no noise parameter is sampled.  This works for *any* forward
        model, linear or nonlinear:

        log p(y|θ) = -(α + n/2) · log(β + 0.5·||y - f(θ)||²) + const

        The posterior mean of σ² is ``(β + 0.5·RSS) / (α + n/2 - 1)``
        and can be retrieved via ``posterior_sigma(theta)``.

    Parameters
    ----------
    forward_model : callable
        ``f(theta_model) -> array-like, shape (n_obs,)``.
    y_obs : array-like, shape (n_obs,)
        Observed data.
    noise_std : float or None
        Fixed σ.  Use ``None`` for either estimated or marginalised modes.
    marginalise_noise : bool
        If ``True`` (and ``noise_std=None``), analytically marginalise σ²
        using an Inverse-Gamma prior.  Default ``False``.
    inv_gamma_alpha : float
        Shape parameter α of the Inverse-Gamma prior on σ².  Default 1.0
        (weakly informative).  Only used when ``marginalise_noise=True``.
    inv_gamma_beta : float
        Scale parameter β of the Inverse-Gamma prior on σ².  Default
        ``1e-4`` (weakly informative).  Only used when
        ``marginalise_noise=True``.

    Examples
    --------
    Fixed noise::

        ll = GaussianNoiseLikelihood(forward_model=f, y_obs=y, noise_std=0.05)
        problem = mc.Problem(prior=log_prior, likelihood=ll,
                             param_names=["E", "zeta"])

    Estimated noise (last param = log σ)::

        ll = GaussianNoiseLikelihood(forward_model=f, y_obs=y)
        problem = mc.Problem(prior=log_prior, likelihood=ll,
                             param_names=["E", "zeta", "log_sigma"])

    Marginalised noise (no noise param in theta)::

        ll = GaussianNoiseLikelihood(forward_model=f, y_obs=y,
                                     marginalise_noise=True)
        problem = mc.Problem(prior=log_prior, likelihood=ll,
                             param_names=["E", "zeta"])
    """

    def __init__(self, forward_model, y_obs, noise_std=None,
                 marginalise_noise=False,
                 inv_gamma_alpha=1.0, inv_gamma_beta=1e-4):
        self._forward = forward_model
        self._y_obs = np.asarray(y_obs, dtype=float).ravel()
        self._n_obs = self._y_obs.size

        if noise_std is not None and float(noise_std) <= 0:
            raise ValueError("noise_std must be positive.")
        self._noise_std = float(noise_std) if noise_std is not None else None

        if marginalise_noise and noise_std is not None:
            raise ValueError(
                "marginalise_noise=True is incompatible with a fixed noise_std."
            )
        self._marginalise = marginalise_noise
        self._alpha = float(inv_gamma_alpha)
        self._beta = float(inv_gamma_beta)

    @property
    def estimate_noise(self):
        """True if log σ is a free parameter (last element of theta)."""
        return self._noise_std is None and not self._marginalise

    @property
    def marginalise_noise(self):
        """True if σ² is analytically marginalised out."""
        return self._marginalise

    def __call__(self, theta):
        """Evaluate log p(y | theta).

        Parameters
        ----------
        theta : array-like
            - Fixed/marginalised mode: model parameters only.
            - Estimated mode: model parameters + ``log_sigma`` as last element.

        Returns
        -------
        float
        """
        theta = np.asarray(theta, dtype=float)

        if self.estimate_noise:
            theta_model = theta[:-1]
            sigma = np.exp(float(theta[-1]))
            y_pred = np.asarray(self._forward(theta_model), dtype=float).ravel()
            residuals = self._y_obs - y_pred
            return (-0.5 * np.dot(residuals, residuals) / sigma**2
                    - self._n_obs * np.log(sigma))

        y_pred = np.asarray(self._forward(theta), dtype=float).ravel()
        residuals = self._y_obs - y_pred
        rss = float(np.dot(residuals, residuals))

        if self._marginalise:
            # Marginal likelihood with InvGamma(alpha, beta) prior on sigma^2
            # log p(y|theta) = -(alpha + n/2) * log(beta + 0.5*RSS) + const
            a = self._alpha + 0.5 * self._n_obs
            b = self._beta + 0.5 * rss
            return -a * np.log(b)

        # Fixed sigma
        sigma = self._noise_std
        return -0.5 * rss / sigma**2 - self._n_obs * np.log(sigma)

    def posterior_sigma(self, theta):
        """Posterior mean of σ given model parameters (marginalised mode only).

        Returns sqrt(E[σ² | theta, y]), which can be used as a point estimate
        of the noise level after inference.

        Parameters
        ----------
        theta : array-like
            Model parameters (not including log_sigma).
        """
        if not self._marginalise:
            raise RuntimeError(
                "posterior_sigma is only available in marginalised mode."
            )
        theta = np.asarray(theta, dtype=float)
        y_pred = np.asarray(self._forward(theta), dtype=float).ravel()
        rss = float(np.dot(self._y_obs - y_pred, self._y_obs - y_pred))
        a = self._alpha + 0.5 * self._n_obs
        b = self._beta + 0.5 * rss
        # E[sigma^2] = b / (a - 1)  for a > 1
        if a <= 1:
            raise RuntimeError("alpha + n/2 must be > 1 for a finite posterior mean.")
        return np.sqrt(b / (a - 1))
