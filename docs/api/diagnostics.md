# Diagnostics

Convergence diagnostics for MCMC chains.  All functions operate on raw
``np.ndarray`` sample arrays (shape ``(n_samples, n_params)``) and are also
exposed as methods on :class:`~mcmckit.core.result.Result`.

!!! tip
    For multi-chain workflows use :class:`~mcmckit.core.multichain.MultiChainResult`
    which calls these functions internally and formats the results as a table.

## ess

::: mcmckit.core.diagnostics.ess

---

## autocorr

::: mcmckit.core.diagnostics.autocorr

---

## gelman_rubin

::: mcmckit.core.diagnostics.gelman_rubin

---

## convergence_summary

::: mcmckit.core.diagnostics.convergence_summary
