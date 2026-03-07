# Bayesian Model Averaging

Model comparison selects the *best* model.  **Bayesian Model Averaging (BMA)**
goes further: instead of committing to one model, it combines all models'
predictions weighted by their posterior probability.  The result is a predictive
distribution that accounts for **model uncertainty**, not just parameter
uncertainty.

## When to use BMA

- The evidence is not decisive between two or more models (Bayes factor < 10:1)
- You want predictions that are robust to the choice of model structure
- You are comparing models that are all physically plausible (e.g. different
  boundary conditions, different DOF counts)

When evidence *is* decisive ($w_{\text{best}} \approx 1$), BMA collapses to the
best model — it gives the same answer but with no extra cost.

## Model weights

After running `ModelComparison`, call `weights()` to get posterior model
probabilities:

```python
comp.run()
w = comp.weights()
# array([0.97, 0.03])  — M2 is overwhelmingly preferred
```

With informative prior model probabilities (e.g. expert knowledge):

```python
# sorted_names matches summary() order (best → worst)
w = comp.weights(prior_weights=[0.3, 0.7])  # favour M2 a priori
```

## BMA prediction

```python
bma = comp.predict(
    forward_models={
        "M1: single k":  lambda theta: natural_frequencies(theta[0], theta[0]),
        "M2: k1 + k2":   lambda theta: natural_frequencies(theta[0], theta[1]),
    },
    n_eval=2000,
)

print(bma)
# BMAResult(n_models=2, n_obs=2, weights=[M2: k1+k2: 0.973, M1: single k: 0.027])

bma.mean()   # BMA mean prediction, shape (n_obs,)
bma.std()    # BMA std, shape (n_obs,)
bma.quantile([0.05, 0.95])  # credible interval

bma.plot_bands(x=freq_indices, y_obs=observed_freqs)
```

## Per-model decomposition

```python
dec = bma.decompose()
# {'M2: k1+k2': {'weight': 0.973, 'mean': array([...]), 'std': array([...])},
#  'M1: single k': {'weight': 0.027, ...}}

bma.plot_decompose()   # line width ∝ model weight
```

This is useful to verify that the dominant model drives the BMA prediction, or
to identify when two models make noticeably different predictions.

## Example output

Running the scenario in `examples/model_averaging.py` (2-DOF frame, M1 vs M2):

```
Model                   log p(y|M)    log10 BF  Evidence vs best
M2: k1 + k2                  4.12        0.00  Barely worth mentioning   <-- best
M1: single k                 2.87       -0.54  Substantial

Posterior model weights:
  M2: k1 + k2          w = 0.947
  M1: single k         w = 0.053
```

M2 is strongly preferred but not decisively — the BMA prediction will be
mostly M2 with a small M1 contribution, slightly widening the credible band.

## API reference

- [`ModelComparison.weights()`](../api/model_comparison.md)
- [`ModelComparison.predict()`](../api/model_comparison.md)
- [`BMAResult`](../api/model_comparison.md)

See the full runnable script at `examples/model_averaging.py`.
