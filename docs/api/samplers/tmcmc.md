# TMCMC

Transitional Markov Chain Monte Carlo (Ching & Chen 2007).

Bridges from the prior to the posterior through a sequence of tempered distributions and accumulates the **log-evidence** (log marginal likelihood) as a by-product.

```python
import numpy as np
import mcmckit as mc

prior_samples = np.random.uniform(-10, 10, size=(1000, 2))

tmcmc  = mc.TMCMC(n_particles=1000, n_mcmc_steps=3, target_ess_ratio=0.5)
result = tmcmc.run(problem, prior_samples=prior_samples)

print(f"log-evidence: {result.log_evidence:.4f}")
print(f"stages:       {tmcmc.stage}")

# particle evolution plot
tmcmc.plot_stages(max_stages=6)
```

!!! note "Prior samples"
    TMCMC cannot draw from the prior automatically.  Pass samples drawn from
    $p(\theta)$ as `prior_samples`.

---

::: mcmckit.samplers.tmcmc.TMCMC
