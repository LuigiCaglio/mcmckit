from __future__ import annotations
from typing import Union
import numpy as np


class GaussianNoiseLikelihood:
    """Log-likelihood for a forward model with Gaussian observation noise.

    Supports **scalar** or **per-channel** noise, where a "channel" is a
    group of observations sharing one noise parameter (e.g. all measurements
    of the 1st natural frequency form one channel, all measurements of the
    2nd natural frequency form another).

    Three noise modes
    -----------------
    **Fixed σ** — ``noise_std`` is a float or array:

        log p(y|θ) = -0.5 · Σᵢ ||rᵢ||² / σᵢ² - nᵢ·log(σᵢ)

    **Estimated σ** — ``noise_std=None``: the last ``n_channels`` elements of
    ``theta`` are ``[log σ₁, …, log σₖ]`` and are sampled jointly with the
    model parameters.

    **Marginalised σ** — ``noise_std=None, marginalise_noise=True``:
    each σᵢ² is given an Inverse-Gamma(αᵢ, βᵢ) prior and integrated out
    analytically.  Works for *any* nonlinear forward model:

        log p(y|θ) = Σᵢ  -(αᵢ + nᵢ/2) · log(βᵢ + 0.5·||rᵢ||²)

    Parameters
    ----------
    forward_model : callable
        ``f(theta_model) -> array-like, shape (n_obs,)``.
    y_obs : array-like, shape (n_obs,)
        Observed data (flat vector).
    noise_std : float, array-like, or None
        Fixed noise standard deviation(s).  Options:

        - ``float`` — same σ for every observation.
        - ``array of length n_obs`` — per-observation σ.
        - ``array of length n_channels`` — one σ per channel group;
          requires ``groups`` to be set.
        - ``None`` — σ is free (estimated or marginalised).

    groups : list of array-like of int, or None
        Index groups defining channels.  E.g.::

            groups=[[0,1,2,3], [4,5,6,7]]   # 2 channels of 4 obs each

        ``None`` (default) treats all observations as one channel.
        When ``noise_std`` is a per-channel array or ``None``, ``groups``
        is required.

    marginalise_noise : bool
        If ``True`` (and ``noise_std=None``), analytically marginalise each
        σᵢ² using independent Inverse-Gamma priors.  Default ``False``.

    inv_gamma_alpha : float or array-like
        Shape parameter(s) α of the Inverse-Gamma prior on σᵢ².
        Scalar → same for all channels.  Default 1.0.

    inv_gamma_beta : float or array-like
        Scale parameter(s) β of the Inverse-Gamma prior on σᵢ².
        Scalar → same for all channels.  Default 1e-4.

    Examples
    --------
    Scalar fixed noise::

        ll = GaussianNoiseLikelihood(forward_model=f, y_obs=y, noise_std=0.05)

    Per-channel fixed noise (2 frequencies, 4 repetitions each)::

        ll = GaussianNoiseLikelihood(
            forward_model=f, y_obs=y,
            noise_std=[0.05, 0.10],
            groups=[[0,1,2,3], [4,5,6,7]],
        )

    Per-channel estimated noise (2 free log_σ params appended to theta)::

        ll = GaussianNoiseLikelihood(
            forward_model=f, y_obs=y,
            groups=[[0,1,2,3], [4,5,6,7]],
        )
        # theta = [*model_params, log_sigma_1, log_sigma_2]

    Per-channel marginalised noise::

        ll = GaussianNoiseLikelihood(
            forward_model=f, y_obs=y,
            groups=[[0,1,2,3], [4,5,6,7]],
            marginalise_noise=True,
            inv_gamma_alpha=[2.0, 2.0],
            inv_gamma_beta=[0.05**2, 0.10**2],
        )
    """

    def __init__(self, forward_model, y_obs,
                 noise_std: Union[float, list, np.ndarray, None] = None,
                 groups=None,
                 marginalise_noise: bool = False,
                 inv_gamma_alpha: Union[float, list, np.ndarray] = 1.0,
                 inv_gamma_beta: Union[float, list, np.ndarray] = 1e-4):

        self._forward = forward_model
        self._y_obs = np.asarray(y_obs, dtype=float).ravel()
        n_obs = self._y_obs.size

        # --- resolve groups --------------------------------------------------
        if groups is None:
            self._groups = [np.arange(n_obs)]
        else:
            self._groups = [np.asarray(g, dtype=int) for g in groups]
            # validate
            all_idx = np.concatenate(self._groups)
            if len(all_idx) != n_obs or set(all_idx.tolist()) != set(range(n_obs)):
                raise ValueError(
                    "groups must partition range(n_obs) exactly once."
                )
        n_channels = len(self._groups)

        # --- resolve noise_std -----------------------------------------------
        if marginalise_noise and noise_std is not None:
            raise ValueError(
                "marginalise_noise=True is incompatible with a fixed noise_std."
            )
        self._marginalise = marginalise_noise

        if noise_std is None:
            self._sigmas = None          # free or marginalised
        else:
            arr = np.asarray(noise_std, dtype=float).ravel()
            if arr.size == 1:
                # broadcast scalar to all obs
                self._sigmas = np.full(n_obs, float(arr[0]))
            elif arr.size == n_channels:
                # one sigma per channel → broadcast to per-obs
                self._sigmas = np.empty(n_obs)
                for sigma_i, idx in zip(arr, self._groups):
                    self._sigmas[idx] = sigma_i
            elif arr.size == n_obs:
                self._sigmas = arr
            else:
                raise ValueError(
                    f"noise_std has {arr.size} elements; expected 1, "
                    f"{n_channels} (n_channels), or {n_obs} (n_obs)."
                )
            if np.any(self._sigmas <= 0):
                raise ValueError("All noise_std values must be positive.")

        self._n_channels = n_channels

        # --- InvGamma hyperparameters ----------------------------------------
        alpha_arr = np.asarray(inv_gamma_alpha, dtype=float).ravel()
        beta_arr  = np.asarray(inv_gamma_beta,  dtype=float).ravel()
        self._alpha = np.broadcast_to(alpha_arr, (n_channels,)).copy()
        self._beta  = np.broadcast_to(beta_arr,  (n_channels,)).copy()

    # -------------------------------------------------------------------------

    @property
    def estimate_noise(self):
        """True if log σᵢ are free parameters (last n_channels elements of theta)."""
        return self._sigmas is None and not self._marginalise

    @property
    def marginalise_noise(self):
        """True if σᵢ² are analytically marginalised out."""
        return self._marginalise

    @property
    def n_noise_params(self):
        """Number of free noise parameters appended to theta (0 if fixed/marginalised)."""
        return self._n_channels if self.estimate_noise else 0

    def __call__(self, theta):
        """Evaluate log p(y | theta).

        Parameters
        ----------
        theta : array-like
            - Fixed / marginalised: model parameters only.
            - Estimated: model parameters followed by ``n_noise_params``
              values of ``log σᵢ``.

        Returns
        -------
        float
        """
        theta = np.asarray(theta, dtype=float)

        if self.estimate_noise:
            theta_model = theta[:-self._n_channels]
            log_sigmas   = theta[-self._n_channels:]
            sigmas = np.exp(log_sigmas)
            # broadcast to per-obs
            sigma_arr = np.empty(self._y_obs.size)
            for s, idx in zip(sigmas, self._groups):
                sigma_arr[idx] = s
        elif self._sigmas is not None:
            theta_model = theta
            sigma_arr   = self._sigmas
        else:
            theta_model = theta
            sigma_arr   = None   # marginalised

        y_pred = np.asarray(self._forward(theta_model), dtype=float).ravel()
        residuals = self._y_obs - y_pred

        if self._marginalise:
            ll = 0.0
            for k, idx in enumerate(self._groups):
                r = residuals[idx]
                rss = float(np.dot(r, r))
                n_k = len(idx)
                a = self._alpha[k] + 0.5 * n_k
                b = self._beta[k]  + 0.5 * rss
                ll -= a * np.log(b)
            return ll

        # fixed or estimated
        # Both residual factors must be scaled: the exponent is
        # sum(r_i^2 / sigma_i^2), not sum(r_i^2 / sigma_i).
        scaled = residuals / sigma_arr
        return float(-0.5 * np.dot(scaled, scaled) - np.sum(np.log(sigma_arr)))

    def posterior_sigma(self, theta):
        """Posterior mean of each σᵢ given model parameters (marginalised mode).

        Parameters
        ----------
        theta : array-like
            Model parameters only.

        Returns
        -------
        np.ndarray, shape (n_channels,)
            ``sqrt(E[σᵢ² | theta, y])`` for each channel.
        """
        if not self._marginalise:
            raise RuntimeError(
                "posterior_sigma is only available in marginalised mode."
            )
        theta = np.asarray(theta, dtype=float)
        y_pred = np.asarray(self._forward(theta), dtype=float).ravel()
        residuals = self._y_obs - y_pred

        result = np.empty(self._n_channels)
        for k, idx in enumerate(self._groups):
            r = residuals[idx]
            rss = float(np.dot(r, r))
            n_k = len(idx)
            a = self._alpha[k] + 0.5 * n_k
            b = self._beta[k]  + 0.5 * rss
            if a <= 1:
                raise RuntimeError(
                    f"alpha[{k}] + n_k/2 must be > 1 for a finite posterior mean."
                )
            result[k] = np.sqrt(b / (a - 1))
        return result
