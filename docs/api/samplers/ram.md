# RAM

Robust Adaptive Metropolis (Vihola 2012).  Self-tunes the proposal covariance via a rank-1 Cholesky update, targeting a 23.4% acceptance rate.

```python
import numpy as np
import mcmckit as mc

sampler = mc.RAM(
    n_samples=15_000,
    initial_cov=np.eye(2) * 0.01,   # can start far from optimal
)
result = sampler.run(problem, x0=[0.0, 0.0])

print(sampler.proposal_cov)   # adapted covariance
```

---

::: mcmckit.samplers.ram.RAM
