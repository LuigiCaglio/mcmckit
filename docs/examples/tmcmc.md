# TMCMC

Source: [`examples/tmcmc_and_gibbs.py`](https://github.com/LuigiCaglio/mcmckit/blob/master/examples/tmcmc_and_gibbs.py)

TMCMC is the recommended sampler when you need:

- A **log-evidence** estimate (model comparison, Bayes factors)
- Sampling from a **multimodal** posterior
- The prior and posterior are very different (large information gain)

---

## Basic usage

TMCMC requires prior samples drawn manually:

```python
import numpy as np
import mcmckit as mc

# draw N prior samples
N = 1000
prior_samples = np.random.uniform(-10, 10, size=(N, 2))

tmcmc  = mc.TMCMC(n_particles=N, n_mcmc_steps=3)
result = tmcmc.run(problem, prior_samples=prior_samples)

print(f"log-evidence : {result.log_evidence:.4f}")
print(f"stages       : {tmcmc.stage}")
print(f"mean         : {result.mean()}")
```

---

## Stage-by-stage execution

Inspect the β progression in real time:

```python
tmcmc = mc.TMCMC(n_particles=N, n_mcmc_steps=3)
tmcmc.initialize_with_samples(problem, prior_samples)

while tmcmc.beta < 1.0:
    tmcmc.run_stage()
    print(f"stage {tmcmc.stage:2d} | β={tmcmc.beta:.4f} | log_ev={tmcmc.log_evidence:.3f}")

result = tmcmc.get_result()
```

---

## Particle evolution plot

Visualise how particles migrate from the prior to the posterior:

```python
tmcmc.plot_stages(
    max_stages=6,      # subsample to 6 stages for clarity
    levels=4,
    title="Prior → Posterior",
)
```

Each stage is drawn with KDE contours (lower triangle) and 1-D KDE curves (diagonal), coloured from light (prior, β=0) to dark (posterior, β=1).

---

## Parallel likelihood evaluation

For expensive forward models, parallelise over particles:

```python
tmcmc = mc.TMCMC(n_particles=500, n_mcmc_steps=3, n_workers=8)
result = tmcmc.run(problem, prior_samples=prior_samples)
```

This uses Python's `ProcessPoolExecutor` internally, so the likelihood function must be picklable (i.e. defined at module level, not as a lambda or nested function).
