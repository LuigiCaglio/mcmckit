# AdaptiveMALA

MALA with automatic step-size tuning via log-space adaptation targeting 57.4% acceptance.

Requires `grad_log_likelihood` (and optionally `grad_log_prior`) on the `Problem`.

```python
import mcmckit as mc

sampler = mc.AdaptiveMALA(
    n_samples=15_000,
    initial_step_size=0.05,   # can start far from optimal
    target_rate=0.574,
    gamma=0.6,
)
result = sampler.run(problem_with_grad, x0=[0.0, 0.0])

print(f"final step size: {sampler.step_size:.4f}")
```

---

::: mcmckit.samplers.adaptive_mala.AdaptiveMALA
