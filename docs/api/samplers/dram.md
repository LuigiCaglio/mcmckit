# DRAM

Delayed Rejection Adaptive Metropolis (Haario et al. 2006).

Combines:

- **Adaptive Metropolis (AM):** empirical covariance adaptation.
- **Delayed Rejection (DR):** when the first proposal is rejected, try a second, smaller proposal with a Tierney-Mira corrected acceptance probability.

DRAM is the recommended general-purpose adaptive sampler — it is robust to poor initialisation and handles correlated posteriors well.

```python
import numpy as np
import mcmckit as mc

sampler = mc.DRAM(
    n_samples=15_000,
    initial_cov=np.eye(2) * 0.5,
    dr_scale=0.1,        # second proposal = dr_scale × first
    adapt_start=200,     # start adapting after this many steps
)
result = sampler.run(problem, x0=[0.0, 0.0])

print(f"stage-1 acc: {sampler.stage1_acceptance_rate:.3f}")
print(f"stage-2 acc: {sampler.stage2_acceptance_rate:.3f}")
```

---

::: mcmckit.samplers.dram.DRAM
