# Full-run helpers

Thin loops over the [step functions](steps.md), for when you do not need
control of the recursion. Both interfaces run identical code.

```python
result = mc.ram(log_post, x0=[0.0, 0.0], n_samples=10_000)

print(result.mean(), result.std())
result.discard(1000).plot_corner()
```

Each helper accepts either a bare `log_post` callable or a
[`Problem`](problem.md), and returns a [`Result`](result.md).

---

::: mcmckit.runners.metropolis

---

::: mcmckit.runners.ram

---

::: mcmckit.runners.dram

---

::: mcmckit.runners.mala

---

::: mcmckit.runners.adaptive_mala

---

::: mcmckit.runners.gibbs
