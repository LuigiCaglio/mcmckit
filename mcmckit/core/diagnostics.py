"""MCMC convergence diagnostics.

Standalone functions that operate on raw sample arrays.  These are also
exposed as methods on :class:`~mcmckit.core.result.Result` for convenience.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Autocorrelation
# ------------------------------------------------------------------

def autocorr(samples: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Normalised autocorrelation function for each parameter.

    Computed via FFT for efficiency (O(N log N) per parameter).

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, n_params)
    max_lag : int
        Maximum lag to return.  Clipped to n_samples - 1.

    Returns
    -------
    np.ndarray, shape (max_lag + 1, n_params)
        ``acf[k, i]`` is the autocorrelation at lag ``k`` for parameter ``i``.
        ``acf[0, :] == 1`` by definition.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if samples.ndim == 1:
        samples = samples[:, None]
    n, d = samples.shape
    max_lag = min(max_lag, n - 1)

    acf = np.empty((max_lag + 1, d))
    for i in range(d):
        x = samples[:, i] - samples[:, i].mean()
        # FFT-based circular autocorrelation, then take first n lags
        f = np.fft.rfft(x, n=2 * n)
        power = (f * np.conj(f)).real
        raw = np.fft.irfft(power)[:n]
        raw /= raw[0]  # normalise so lag-0 == 1
        acf[:, i] = raw[:max_lag + 1]
    return acf


# ------------------------------------------------------------------
# Effective sample size
# ------------------------------------------------------------------

def ess(samples: np.ndarray) -> np.ndarray:
    """Effective sample size (ESS) per parameter.

    Uses Geyer's (1992) initial positive sequence estimator: accumulate
    pairs of autocorrelations as long as their sum is positive.  This
    avoids the noise blow-up of naively summing all lags.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, n_params)

    Returns
    -------
    np.ndarray, shape (n_params,)
        ESS for each parameter, capped at n_samples.

    References
    ----------
    Geyer, C. J. (1992). Practical Markov Chain Monte Carlo.
    Statistical Science, 7(4), 473–483.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if samples.ndim == 1:
        samples = samples[:, None]
    n, d = samples.shape
    max_lag = min(n - 1, 500)
    acf = autocorr(samples, max_lag=max_lag)  # (max_lag+1, d)

    result = np.empty(d)
    for i in range(d):
        rho = acf[:, i]
        # Sum pairs rho[2k] + rho[2k+1] while positive (Geyer's criterion)
        gamma_sum = 0.0
        for k in range(1, (len(rho) - 1) // 2 + 1):
            pair = rho[2 * k] + rho[2 * k + 1] if 2 * k + 1 < len(rho) else rho[2 * k]
            if pair <= 0:
                break
            gamma_sum += pair
        tau = -1.0 + 2.0 * gamma_sum  # integrated autocorrelation time
        result[i] = min(n / max(1.0 + tau, 1.0), float(n))
    return result


# ------------------------------------------------------------------
# Gelman-Rubin R-hat
# ------------------------------------------------------------------

def gelman_rubin(chains: list) -> np.ndarray:
    """Split-chain Gelman-Rubin :math:`\\hat{R}` statistic.

    Each chain is split in half before computing :math:`\\hat{R}`, which
    allows detection of non-stationarity within a single chain.
    Values close to 1.0 indicate convergence; a common threshold is
    :math:`\\hat{R} < 1.01` (strict) or :math:`< 1.1` (permissive).

    Parameters
    ----------
    chains : list of np.ndarray, each shape (n_samples, n_params)
        Or a list of :class:`~mcmckit.core.result.Result` objects — their
        ``.samples`` attribute is used automatically.

    Returns
    -------
    np.ndarray, shape (n_params,)
        :math:`\\hat{R}` for each parameter.

    Raises
    ------
    ValueError
        If fewer than 2 chains are provided or chains are too short to split.

    References
    ----------
    Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation
    using multiple sequences. Statistical Science, 7(4), 457–472.

    Vehtari, A., et al. (2021). Rank-normalization, folding, and localization:
    An improved Rhat for assessing convergence of MCMC. Bayesian Analysis,
    16(2), 667–718.
    """
    # Accept Result objects or raw arrays
    raw = []
    for c in chains:
        arr = c.samples if hasattr(c, "samples") else np.asarray(c, dtype=float)
        raw.append(arr)

    if len(raw) < 2:
        raise ValueError("gelman_rubin requires at least 2 chains.")

    # Split each chain in half → 2*M sub-chains
    split = []
    for arr in raw:
        n = len(arr)
        if n < 4:
            raise ValueError("Each chain must have at least 4 samples to split.")
        half = n // 2
        split.append(arr[:half])
        split.append(arr[half: 2 * half])

    M = len(split)              # number of sub-chains
    N = len(split[0])           # length of each sub-chain
    d = split[0].shape[1]

    chain_means = np.array([s.mean(axis=0) for s in split])   # (M, d)
    grand_mean  = chain_means.mean(axis=0)                     # (d,)

    # Between-chain variance B
    B = N / (M - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)

    # Within-chain variance W
    chain_vars = np.array([s.var(axis=0, ddof=1) for s in split])  # (M, d)
    W = chain_vars.mean(axis=0)

    # Pooled variance estimate
    var_hat = (N - 1) / N * W + B / N

    # Avoid division by zero for degenerate parameters
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.where(W > 0, np.sqrt(var_hat / W), np.nan)

    return rhat


# ------------------------------------------------------------------
# Convenience summary
# ------------------------------------------------------------------

def convergence_summary(chains: list, threshold_rhat: float = 1.01) -> dict:
    """Compute R-hat and per-chain ESS, return a summary dict.

    Parameters
    ----------
    chains : list of Result or np.ndarray
    threshold_rhat : float
        Warn if any R-hat exceeds this value.  Default 1.01.

    Returns
    -------
    dict with keys ``rhat`` (array), ``ess`` (list of arrays),
    ``converged`` (bool), ``warnings`` (list of str).
    """
    rhat = gelman_rubin(chains)
    ess_per_chain = [
        ess(c.samples if hasattr(c, "samples") else c) for c in chains
    ]
    warnings_list = []
    for i, r in enumerate(rhat):
        if np.isnan(r):
            warnings_list.append(f"param {i}: R-hat is NaN (constant chain?)")
        elif r > threshold_rhat:
            warnings_list.append(
                f"param {i}: R-hat = {r:.4f} > {threshold_rhat} (not converged)"
            )
    return {
        "rhat": rhat,
        "ess": ess_per_chain,
        "converged": len(warnings_list) == 0,
        "warnings": warnings_list,
    }
