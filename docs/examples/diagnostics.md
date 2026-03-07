# Convergence diagnostics & multi-chain

Reliable Bayesian inference requires verifying that MCMC chains have converged
and that the effective sample size is sufficient.  mcmckit provides:

- **ESS** — effective sample size per parameter (how many independent samples the chain is worth)
- **Autocorrelation** — lag-ACF plots to diagnose poor mixing
- **Gelman-Rubin $\hat{R}$** — multi-chain convergence statistic (target: < 1.01)

## Running multiple chains

The easiest entry point is `run_chains`, which handles starting-point jitter,
deep-copies the sampler for each chain, and returns a `MultiChainResult`:

```python
import numpy as np
import mcmckit as mc

# Define your problem as usual
problem = mc.Problem(prior=log_prior, likelihood=likelihood, param_names=["k1", "k2"])

sampler = mc.DRAM(n_samples=10_000, initial_cov=0.5 * np.eye(2))

mc_result = mc.run_chains(sampler, problem, x0=[10.0, 8.0], n_chains=4)
```

## Convergence summary

```python
mc_result.summary(discard=2000)
```

Example output:

```
Parameter           R-hat    ESS (total)        Mean        Std
------------------------------------------------------------------
k1                 1.0012           3241      10.023       0.412
k2                 1.0008           3187       7.981       0.389

  All R-hat < 1.01 — chains appear converged.
```

- **R-hat close to 1.0** — chains agree on the same distribution.
- **High ESS** — the pooled chain contains many effectively independent samples.
- A `*` flag appears next to any parameter where $\hat{R} > 1.01$.

## Pool chains for downstream analysis

```python
pooled = mc_result.pool(discard=2000)
pooled.plot_corner()
```

## Trace plots

```python
mc_result.plot_traces(title="4 chains — DRAM")
```

Overlaid traces from all chains in different colours.  Chains that mix well
look like overlapping fuzzy caterpillars sharing the same vertical range.

## Single-chain diagnostics

On any `Result` object:

```python
result = sampler.run(problem, x0=[10.0, 8.0]).discard(2000)

# Effective sample size
print(result.ess())
# [3102.  2987.]

# Autocorrelation plot
result.plot_autocorr(max_lag=80)
```

## Standalone functions

```python
# From a list of Result objects (or raw arrays)
rhat = mc.gelman_rubin([result_chain1, result_chain2, result_chain3, result_chain4])

# Summary dict with warnings
diag = mc.convergence_summary([r1, r2, r3, r4], threshold_rhat=1.01)
print(diag["converged"])    # True / False
print(diag["warnings"])     # list of strings
```

## Rule of thumb

| $\hat{R}$ | Interpretation |
|---|---|
| < 1.01 | Converged (strict) |
| 1.01 – 1.1 | Probably OK, run longer to be safe |
| > 1.1 | Not converged — discard more burn-in or run longer |

ESS > 400 per parameter is generally sufficient for stable posterior summaries.
For tail quantiles or credible intervals, aim for ESS > 1000.
