# MetropolisHastings

Random-walk Metropolis-Hastings with a fixed Gaussian proposal.

```python
import numpy as np
import mcmckit as mc

sampler = mc.MetropolisHastings(
    proposal_cov=np.eye(2) * 0.5,
    n_samples=10_000,
)
result = sampler.run(problem, x0=[0.0, 0.0])
```

`proposal_cov` accepts a scalar (isotropic), a 1-D array (diagonal), or a 2-D array (full).

---

::: mcmckit.samplers.metropolis.MetropolisHastings
