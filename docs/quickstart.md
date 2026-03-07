# Quickstart

## Installation

```bash
pip install mcmckit[plot]
```

## 1. Define a Problem

`Problem` wraps your prior and likelihood as plain Python callables:

```python
import numpy as np
import mcmckit as mc

def log_prior(theta):
    # flat prior on [-10, 10]^2
    if np.any(np.abs(theta) > 10):
        return -np.inf
    return 0.0

def log_likelihood(theta):
    # 2-D correlated Gaussian centred at (2, -1)
    true_mean = np.array([2.0, -1.0])
    true_prec = np.linalg.inv([[1.0, 0.8], [0.8, 1.0]])
    diff = theta - true_mean
    return -0.5 * diff @ true_prec @ diff

problem = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["x", "y"],
)
```

## 2. Run a sampler

All samplers expose `.run(problem, x0)`:

```python
sampler = mc.DRAM(n_samples=10_000, initial_cov=np.eye(2) * 0.5)
result  = sampler.run(problem, x0=[0.0, 0.0])
```

## 3. Discard burn-in and inspect

```python
result = result.discard(2000)

print(result.mean())   # → [~2.0, ~-1.0]
print(result.std())
print(result.acceptance_rate)

samples = result.samples   # raw numpy array, shape (8000, 2)
```

## 4. Plot

```python
result.plot_corner(true_values=[2.0, -1.0], title="Posterior")
result.plot_trace()
result.plot_marginals()
```

Corner plot styles: `"corner"` (default), `"scatter"`, `"full"`, `"kde"`.

## Step-by-step execution

Every sampler supports stopping mid-chain:

```python
sampler = mc.MetropolisHastings(proposal_cov=np.eye(2), n_samples=10_000)
sampler.initialize(problem, x0=[0.0, 0.0])

for i in range(10_000):
    sampler.step()
    if i % 1000 == 0:
        r = sampler.get_result()
        print(f"step {i}: mean = {r.mean()}")
```

## Gradient-based samplers (MALA, AdaptiveMALA)

Provide `grad_log_likelihood` and `grad_log_prior` on the `Problem`:

```python
def grad_log_likelihood(theta):
    return -true_prec @ (theta - true_mean)

def grad_log_prior(theta):
    return np.zeros_like(theta)

problem = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    grad_log_likelihood=grad_log_likelihood,
    grad_log_prior=grad_log_prior,
    param_names=["x", "y"],
)

sampler = mc.AdaptiveMALA(n_samples=10_000, initial_step_size=0.1)
result  = sampler.run(problem, x0=[0.0, 0.0])
```

## TMCMC

TMCMC requires prior samples drawn by the user:

```python
prior_samples = np.random.uniform(-10, 10, size=(1000, 2))

tmcmc  = mc.TMCMC(n_particles=1000, n_mcmc_steps=3)
result = tmcmc.run(problem, prior_samples=prior_samples)

print(f"log-evidence: {result.log_evidence:.3f}")
tmcmc.plot_stages(max_stages=6, title="Prior → Posterior")
```

## Noise estimation

When your likelihood comes from a forward model with unknown measurement noise:

```python
ll = mc.GaussianNoiseLikelihood(
    forward_model=my_fem_model,   # f(theta) -> y_pred
    y_obs=measurements,
    marginalise_noise=True,       # analytically integrate out sigma
)

problem = mc.Problem(prior=log_prior, likelihood=ll,
                     param_names=["E", "zeta"])

result = mc.DRAM(n_samples=10_000, initial_cov=np.eye(2)).run(problem, x0=[...])

# posterior point estimate of sigma
sigma_est = ll.posterior_sigma(result.mean())
```

See [Theory](theory.md) and [Noise estimation](examples/noise.md) for details.
