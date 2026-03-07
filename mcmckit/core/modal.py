from __future__ import annotations
from typing import Union, Optional
import numpy as np


class ModalLikelihood:
    """Log-likelihood for Bayesian model updating with modal data (frequencies + mode shapes).

    Combines a Gaussian likelihood on natural frequency residuals with a
    MAC-based (Modal Assurance Criterion) likelihood on mode shapes.  Both
    formulations follow the derivation of Vanik et al. (1999) as reviewed in
    Kiran et al. (2025, §4.3.1) and the MAC-explicit form of Argyris et al.
    (2020).

    Frequency log-likelihood (per mode r)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **Relative error** (default, dimensionally consistent)::

        ln p(ω̂ᵣ | θ) = -½ · (ω̂ᵣ/ωᵣ(θ) - 1)² / σ²_freq,r

    **Absolute error**::

        ln p(ω̂ᵣ | θ) = -½ · (ω̂ᵣ - ωᵣ(θ))² / σ²_freq,r

    Mode-shape log-likelihood (per mode r)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Based on the projection matrix identity — equivalent to using the MAC::

        ln p(φ̂ᵣ | θ) = -(1 - MACᵣ) / (2 σ²_mac,r)

    where::

        MAC(a, b) = (aᵀb)² / (‖a‖² ‖b‖²)  ∈ [0, 1]

    Total log-likelihood
    ~~~~~~~~~~~~~~~~~~~~
    ::

        ln p(D | θ) = Σᵣ [ ln p(ω̂ᵣ | θ) + ln p(φ̂ᵣ | θ) ]

    Parameters
    ----------
    forward_model : callable
        ``f(theta) -> (freq_pred, mode_shapes_pred)`` where

        - ``freq_pred``: shape ``(n_pred,)`` — predicted natural frequencies
          (same units as ``freq_obs``).
        - ``mode_shapes_pred``: shape ``(n_dof, n_pred)`` — predicted mode
          shapes, one column per mode.  Full model DOF order; ``sensor_dofs``
          selects the measured subset.

    freq_obs : array-like, shape (n_modes,)
        Observed natural frequencies (rad/s or Hz — consistent with forward
        model).

    mode_shapes_obs : array-like, shape (n_sensor_dof, n_modes)
        Observed mode shapes at the measured DOFs only.  One column per mode.
        Scaling and sign are arbitrary (MAC is invariant to both).

    sigma_freq : float, array-like of shape (n_modes,), or None
        Noise standard deviation(s) for the frequency likelihood.

        - ``float`` — same σ for every mode.
        - ``array`` — one σ per mode.
        - ``None`` (default) — σ is a free parameter; the last ``n_modes``
          elements of ``theta`` are treated as ``log σ_freq,r`` and sampled
          jointly with the model parameters.

    sigma_mac : float or array-like of shape (n_modes,)
        Effective noise on the mode shape MAC term.  Controls sensitivity to
        MAC < 1.  Typical values: 0.05–0.20.  Default 0.10.

    sensor_dofs : array-like of int or None
        Zero-based DOF indices used to extract the measured subset from the
        full predicted mode shape matrix.  If ``None``, the forward model is
        assumed to already return mode shapes at the measured DOFs only.

    auto_pair : bool
        If ``True`` (default), automatically pair predicted modes to observed
        modes by solving a linear assignment problem (maximise MAC).  This
        handles mode order switching during MCMC sampling.  Requires
        ``scipy``.

    freq_error : {'relative', 'absolute'}
        How to compute the frequency residual.  ``'relative'`` (default)
        uses ``ω̂/ω(θ) - 1`` which is dimensionally consistent and avoids
        scale sensitivity.

    Examples
    --------
    Minimal usage — forward model returns (freqs, shapes) at sensor DOFs::

        def forward(theta):
            k = theta[0]
            freq = np.array([np.sqrt(k / m)])       # shape (1,)
            shape = np.array([[1.0]])                # shape (1, 1)
            return freq, shape

        ll = mc.ModalLikelihood(
            forward_model=forward,
            freq_obs=np.array([2.34]),
            mode_shapes_obs=np.array([[1.0]]),
            sigma_freq=0.02,
            sigma_mac=0.05,
        )
        problem = mc.Problem(prior=log_prior, likelihood=ll)

    With sensor DOF selection and free frequency noise::

        ll = mc.ModalLikelihood(
            forward_model=forward_full_dof,  # returns (n_freqs, n_dof × n_modes)
            freq_obs=obs_freqs,
            mode_shapes_obs=obs_shapes,      # shape (n_sensors, n_modes)
            sigma_freq=None,                 # free — append log_sigma to theta
            sigma_mac=0.10,
            sensor_dofs=[0, 2, 4],           # measure DOFs 0, 2, 4
        )
        # theta = [*model_params, log_sigma_freq_0, log_sigma_freq_1, ...]
        # problem.n_params = n_model_params + ll.n_noise_params
    """

    def __init__(
        self,
        forward_model,
        freq_obs,
        mode_shapes_obs,
        sigma_freq: Union[float, list, np.ndarray, None] = None,
        sigma_mac: Union[float, list, np.ndarray] = 0.10,
        sensor_dofs: Optional[list] = None,
        auto_pair: bool = True,
        freq_error: str = "relative",
    ):
        self._forward = forward_model

        self._freq_obs = np.asarray(freq_obs, dtype=float).ravel()
        self._n_modes = self._freq_obs.size

        self._shapes_obs = np.asarray(mode_shapes_obs, dtype=float)
        if self._shapes_obs.ndim == 1:
            self._shapes_obs = self._shapes_obs[:, np.newaxis]
        if self._shapes_obs.shape[1] != self._n_modes:
            raise ValueError(
                f"mode_shapes_obs has {self._shapes_obs.shape[1]} columns but "
                f"freq_obs has {self._n_modes} entries."
            )

        self._sensor_dofs = (
            None if sensor_dofs is None else np.asarray(sensor_dofs, dtype=int)
        )
        self._auto_pair = auto_pair

        if freq_error not in ("relative", "absolute"):
            raise ValueError("freq_error must be 'relative' or 'absolute'.")
        self._freq_error = freq_error

        # --- sigma_freq ---------------------------------------------------------
        if sigma_freq is None:
            self._sigma_freq = None        # free parameters
        else:
            arr = np.asarray(sigma_freq, dtype=float).ravel()
            if arr.size == 1:
                self._sigma_freq = np.full(self._n_modes, float(arr[0]))
            elif arr.size == self._n_modes:
                self._sigma_freq = arr.copy()
            else:
                raise ValueError(
                    f"sigma_freq has {arr.size} elements; expected 1 or "
                    f"{self._n_modes} (n_modes)."
                )
            if np.any(self._sigma_freq <= 0):
                raise ValueError("All sigma_freq values must be positive.")

        # --- sigma_mac ----------------------------------------------------------
        arr_mac = np.asarray(sigma_mac, dtype=float).ravel()
        if arr_mac.size == 1:
            self._sigma_mac = np.full(self._n_modes, float(arr_mac[0]))
        elif arr_mac.size == self._n_modes:
            self._sigma_mac = arr_mac.copy()
        else:
            raise ValueError(
                f"sigma_mac has {arr_mac.size} elements; expected 1 or "
                f"{self._n_modes} (n_modes)."
            )
        if np.any(self._sigma_mac <= 0):
            raise ValueError("All sigma_mac values must be positive.")

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        """Number of observed modes."""
        return self._n_modes

    @property
    def n_noise_params(self) -> int:
        """Number of free log-σ_freq parameters appended to theta (0 if fixed)."""
        return self._n_modes if self._sigma_freq is None else 0

    @property
    def estimate_noise(self) -> bool:
        """True if frequency noise parameters are free (last n_modes elements of theta)."""
        return self._sigma_freq is None

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def __call__(self, theta) -> float:
        """Evaluate log p(D | theta).

        Parameters
        ----------
        theta : array-like
            If ``sigma_freq=None``, the last ``n_modes`` elements are
            ``[log σ_freq,0, ..., log σ_freq,n_modes-1]`` and the remaining
            elements are passed to the forward model.  Otherwise ``theta`` is
            passed to the forward model as-is.

        Returns
        -------
        float
        """
        theta = np.asarray(theta, dtype=float)

        if self.estimate_noise:
            theta_model = theta[: -self._n_modes]
            sigma_freq = np.exp(theta[-self._n_modes :])
        else:
            theta_model = theta
            sigma_freq = self._sigma_freq

        freq_pred, shapes_pred = self._forward(theta_model)
        freq_pred = np.asarray(freq_pred, dtype=float).ravel()
        shapes_pred = np.asarray(shapes_pred, dtype=float)
        if shapes_pred.ndim == 1:
            shapes_pred = shapes_pred[:, np.newaxis]

        # Extract sensor DOFs from predicted shapes
        if self._sensor_dofs is not None:
            shapes_pred = shapes_pred[self._sensor_dofs, :]

        # Mode pairing: assign predicted columns to observed modes
        if self._auto_pair:
            col_idx = self._pair_modes(shapes_pred)
            freq_pred = freq_pred[col_idx]
            shapes_pred = shapes_pred[:, col_idx]

        log_lik = 0.0
        for r in range(self._n_modes):
            # --- frequency term ---
            if self._freq_error == "relative":
                if freq_pred[r] <= 0:
                    return -np.inf
                err_f = freq_pred[r] / self._freq_obs[r] - 1.0
            else:
                err_f = freq_pred[r] - self._freq_obs[r]
            log_lik -= 0.5 * err_f ** 2 / sigma_freq[r] ** 2

            # --- MAC term ---
            mac = _mac(self._shapes_obs[:, r], shapes_pred[:, r])
            log_lik -= (1.0 - mac) / (2.0 * self._sigma_mac[r] ** 2)

        return float(log_lik)

    # -------------------------------------------------------------------------
    # Mode pairing
    # -------------------------------------------------------------------------

    def _pair_modes(self, shapes_pred: np.ndarray) -> np.ndarray:
        """Return column indices in shapes_pred that best match each observed mode.

        Uses the Hungarian algorithm (linear_sum_assignment) to maximise the
        sum of MAC values across all mode pairs.

        Parameters
        ----------
        shapes_pred : np.ndarray, shape (n_sensor_dof, n_pred)

        Returns
        -------
        np.ndarray of int, shape (n_modes,)
            ``col_idx[r]`` is the column in ``shapes_pred`` assigned to mode r.
        """
        n_obs = self._n_modes
        n_pred = shapes_pred.shape[1]

        mac_matrix = np.zeros((n_obs, n_pred))
        for i in range(n_obs):
            for j in range(n_pred):
                mac_matrix[i, j] = _mac(self._shapes_obs[:, i], shapes_pred[:, j])

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-mac_matrix)
            # row_ind is always [0, 1, ..., n_obs-1] for a square or tall matrix
            return col_ind
        except ImportError:
            # Greedy fallback if scipy is not available
            assigned = []
            available = list(range(n_pred))
            for i in range(n_obs):
                best = max(available, key=lambda j: mac_matrix[i, j])
                assigned.append(best)
                available.remove(best)
            return np.array(assigned, dtype=int)

    # -------------------------------------------------------------------------
    # Diagnostics helpers
    # -------------------------------------------------------------------------

    def mac_values(self, theta) -> np.ndarray:
        """Compute MAC values for each mode at a given parameter vector.

        Parameters
        ----------
        theta : array-like
            Model parameters (without log-sigma even if estimate_noise=True).

        Returns
        -------
        np.ndarray, shape (n_modes,)
        """
        theta = np.asarray(theta, dtype=float)
        if self.estimate_noise:
            theta_model = theta[: -self._n_modes]
        else:
            theta_model = theta

        freq_pred, shapes_pred = self._forward(theta_model)
        shapes_pred = np.asarray(shapes_pred, dtype=float)
        if shapes_pred.ndim == 1:
            shapes_pred = shapes_pred[:, np.newaxis]
        if self._sensor_dofs is not None:
            shapes_pred = shapes_pred[self._sensor_dofs, :]
        if self._auto_pair:
            col_idx = self._pair_modes(shapes_pred)
            shapes_pred = shapes_pred[:, col_idx]

        return np.array(
            [_mac(self._shapes_obs[:, r], shapes_pred[:, r]) for r in range(self._n_modes)]
        )

    def freq_errors(self, theta) -> np.ndarray:
        """Compute frequency errors (relative or absolute) at a given parameter vector.

        Parameters
        ----------
        theta : array-like
            Model parameters.

        Returns
        -------
        np.ndarray, shape (n_modes,)
        """
        theta = np.asarray(theta, dtype=float)
        if self.estimate_noise:
            theta_model = theta[: -self._n_modes]
        else:
            theta_model = theta

        freq_pred, shapes_pred = self._forward(theta_model)
        freq_pred = np.asarray(freq_pred, dtype=float).ravel()
        shapes_pred = np.asarray(shapes_pred, dtype=float)
        if shapes_pred.ndim == 1:
            shapes_pred = shapes_pred[:, np.newaxis]
        if self._sensor_dofs is not None:
            shapes_pred = shapes_pred[self._sensor_dofs, :]
        if self._auto_pair:
            col_idx = self._pair_modes(shapes_pred)
            freq_pred = freq_pred[col_idx]

        if self._freq_error == "relative":
            return freq_pred / self._freq_obs - 1.0
        return freq_pred - self._freq_obs


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _mac(a: np.ndarray, b: np.ndarray) -> float:
    """Modal Assurance Criterion between two real vectors.

    MAC(a, b) = (aᵀb)² / (‖a‖² · ‖b‖²)

    Invariant to scaling and sign.  Returns 0.0 if either vector is zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = float(np.dot(a, b)) ** 2
    den = float(np.dot(a, a)) * float(np.dot(b, b))
    return num / den if den > 1e-300 else 0.0


def mac_matrix(shapes_a: np.ndarray, shapes_b: np.ndarray) -> np.ndarray:
    """Compute the full MAC matrix between two sets of mode shapes.

    Parameters
    ----------
    shapes_a : np.ndarray, shape (n_dof, n_a)
    shapes_b : np.ndarray, shape (n_dof, n_b)

    Returns
    -------
    np.ndarray, shape (n_a, n_b)
        ``M[i, j] = MAC(shapes_a[:, i], shapes_b[:, j])``.
    """
    shapes_a = np.asarray(shapes_a, dtype=float)
    shapes_b = np.asarray(shapes_b, dtype=float)
    if shapes_a.ndim == 1:
        shapes_a = shapes_a[:, np.newaxis]
    if shapes_b.ndim == 1:
        shapes_b = shapes_b[:, np.newaxis]
    n_a = shapes_a.shape[1]
    n_b = shapes_b.shape[1]
    M = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            M[i, j] = _mac(shapes_a[:, i], shapes_b[:, j])
    return M
