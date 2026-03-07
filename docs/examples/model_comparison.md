# Model class selection

Bayesian model comparison lets you decide **which model structure best explains
the data**, accounting for both goodness-of-fit and model complexity (Occam's
razor is built in).

## Scenario

A 2-DOF shear frame is monitored under ambient vibration.  Two natural
frequencies are measured with Gaussian noise ($\sigma = 0.05$ rad/s).
Three model classes compete:

| Model | Parameters | Description |
|---|---|---|
| M1 | $k$ | 1-DOF — wrong, too simple |
| M2 | $k_1, k_2$ | 2-DOF — correct model |
| M3 | $k_1, k_2, r$ | 2-DOF + free mass ratio — over-parameterised |

True system: $m_1 = m_2 = 1$ kg, $k_1 = 10$ N/m, $k_2 = 8$ N/m.

## How it works

TMCMC estimates the **log marginal likelihood** $\log p(y \mid M)$ as a
byproduct of the tempering process — no extra computation needed.
The **Bayes factor** $B_{12} = p(y \mid M_1) / p(y \mid M_2)$ then directly
quantifies the evidence in favour of one model over another.

A model that fits well *and* uses its prior efficiently gets a high evidence.
M3 has an extra free parameter that doesn't improve the fit, so the prior
volume penalty reduces its evidence — this is Occam's razor.

## Code

```python
import numpy as np
import mcmckit as mc
from scipy.linalg import eigh

# --- True system and data ---
y_obs = np.array([...])   # two observed natural frequencies

# --- Define the three problems ---
lik_m1 = mc.GaussianNoiseLikelihood(fwd_m1, y_obs, noise_std=0.05)
lik_m2 = mc.GaussianNoiseLikelihood(fwd_m2, y_obs, noise_std=0.05)
lik_m3 = mc.GaussianNoiseLikelihood(fwd_m3, y_obs, noise_std=0.05)

problem_m1 = mc.Problem(prior=log_prior_m1, likelihood=lik_m1, param_names=["k"])
problem_m2 = mc.Problem(prior=log_prior_m2, likelihood=lik_m2, param_names=["k1", "k2"])
problem_m3 = mc.Problem(prior=log_prior_m3, likelihood=lik_m3, param_names=["k1", "k2", "r"])

# --- Run comparison ---
comp = mc.ModelComparison(
    models=[
        ("M1: 1-DOF",           problem_m1, prior_m1),
        ("M2: 2-DOF (correct)",  problem_m2, prior_m2),
        ("M3: 2-DOF + mass",    problem_m3, prior_m3),
    ],
    tmcmc_kwargs={"n_particles": 1000},
)
comp.run()
comp.summary()
```

## Output

```
Model                   log p(y|M)    log10 BF  Evidence vs best
-------------------------------------------------------------------------
M2: 2-DOF (correct)          3.237        0.00  Barely worth mentioning   <-- best
M3: 2-DOF + mass             2.327       -0.40  Barely worth mentioning
M1: 1-DOF                  -36.453      -17.24  Decisive
```

### Interpretation

- **M1 vs M2** — log₁₀ BF = 17.24, *Decisive*: the 1-DOF model is overwhelmingly
  rejected.  It cannot reproduce two independent frequencies with a single parameter,
  so its likelihood is systematically low regardless of $k$.

- **M2 vs M3** — log₁₀ BF = 0.40, *Barely worth mentioning*: the correct model
  beats the over-parameterised one, but only weakly (the extra mass parameter
  doesn't hurt the fit much with only 2 observations).  With more data the gap
  grows as the Occam penalty accumulates.

## Pairwise Bayes factors

```python
bf = mc.bayes_factor(res_m2.log_evidence, res_m1.log_evidence)
# {'log_bf': 39.7, 'log10_bf': 17.24, 'bf': 1.6e17,
#  'preferred': 'M1', 'evidence': 'Decisive'}
```

## Posterior of the winning model

```python
res_m2 = comp.get_result("M2: 2-DOF (correct)")
fig = res_m2.plot_corner(true_values=[10.0, 8.0])
```

## Plots

`comp.plot()` produces a two-panel bar chart:

- **Left**: log-evidence for each model
- **Right**: log₁₀ Bayes factor vs the best model, with Jeffreys threshold lines

See the full script at `examples/model_comparison.py`.
