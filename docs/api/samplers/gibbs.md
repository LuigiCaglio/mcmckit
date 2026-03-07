# Gibbs

Metropolis-within-Gibbs sampler.  Updates one parameter block at a time, useful when parameters have different scales or can be tuned independently.

```python
import mcmckit as mc

# Scalar: one param at a time, same proposal std
sampler = mc.Gibbs(n_samples=10_000, proposal_std=0.5)
result  = sampler.run(problem, x0=[0.0, 0.0])

print(sampler.block_acceptance_rates)  # per-block rates

# Block: groups of params, different std per group
sampler = mc.Gibbs(
    n_samples=10_000,
    blocks=[[0, 1], [2, 3]],
    proposal_std=[0.5, 0.2],
)
result = sampler.run(problem_4d, x0=[0, 0, 0, 0])
```

---

::: mcmckit.samplers.gibbs.Gibbs
