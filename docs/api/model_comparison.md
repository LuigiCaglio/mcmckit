# Model comparison

Utilities for Bayesian model class selection based on log-evidence from TMCMC.

!!! note
    Log-evidence is only available from `TMCMC` — it is computed as a byproduct
    of the tempering process.  All other samplers return `result.log_evidence = None`.

## bayes_factor

```python
import mcmckit as mc

bf = mc.bayes_factor(result_1.log_evidence, result_2.log_evidence)
print(bf)
# {'log_bf': 39.7, 'log10_bf': 17.2, 'bf': 1.6e17,
#  'preferred': 'M1', 'evidence': 'Decisive'}
```

::: mcmckit.core.model_comparison.bayes_factor

---

## ModelComparison

```python
comp = mc.ModelComparison(
    models=[
        ("M1: 1-DOF",  problem_1, prior_samples_1),
        ("M2: 2-DOF",  problem_2, prior_samples_2),
    ],
    tmcmc_kwargs={"n_particles": 1000, "n_mcmc_steps": 3},
)
comp.run()
comp.summary()
# Model                   log p(y|M)    log10 BF  Evidence vs best
# -----------------------------------------------------------------------
# M2: 2-DOF                    3.24        0.00  Barely worth mentioning  <-- best
# M1: 1-DOF                  -36.45      -17.24  Decisive
```

::: mcmckit.core.model_comparison.ModelComparison

---

## BMAResult

Returned by `ModelComparison.predict()`.

```python
bma = comp.predict(
    forward_models={"M1": fwd_m1, "M2": fwd_m2},
    n_eval=1000,
)
bma.mean()          # BMA mean prediction
bma.std()           # BMA std
bma.decompose()     # per-model weight, mean, std
bma.plot_bands()    # credible band
bma.plot_decompose() # line width ∝ model weight
```

::: mcmckit.core.model_comparison.BMAResult
