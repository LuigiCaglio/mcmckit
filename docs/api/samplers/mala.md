# MALA

Metropolis-adjusted Langevin Algorithm — gradient-biased proposals for faster mixing.

Requires `grad_log_likelihood` (and optionally `grad_log_prior`) on the `Problem`.

```python
import mcmckit as mc

sampler = mc.MALA(step_size=0.3, n_samples=10_000)
result  = sampler.run(problem_with_grad, x0=[0.0, 0.0])
```

**Optimal acceptance rate** ≈ 57.4% (tune `step_size` accordingly).

---

::: mcmckit.samplers.mala.MALA
