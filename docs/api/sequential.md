# PosteriorPrior

Convert a previous posterior into a prior for the next sequential update step.
Works with **all** samplers — the object implements the same `callable(theta) -> float`
interface as any `log_prior` function.

## Quick start

```python
# Step 1: initial run with a flat prior
result1 = mc.DRAM(n_samples=20_000, initial_cov=np.eye(2)).run(problem1, x0=[10.0, 8.0])

# Step 2: use the posterior as prior — Gaussian approximation
prior2 = result1.as_prior(method='gaussian', discard=2000)

problem2 = mc.Problem(prior=prior2, likelihood=ll_new, param_names=["k1", "k2"])
result2  = mc.DRAM(n_samples=20_000, initial_cov=prior2.cov).run(
    problem2, x0=prior2.mean
)

# Step 3: repeat as new data arrives
prior3 = result2.as_prior(method='gaussian', discard=2000)
```

## Constructing from samples directly

```python
prior = mc.PosteriorPrior(result.discard(2000).samples, method='kde')
```

## Methods

| Method | When to use |
|--------|------------|
| `'gaussian'` | Fast, any dimension. Works well when posterior is unimodal and roughly elliptical — common in structural model updating. |
| `'kde'` | Non-parametric. Handles multimodal / skewed posteriors. Expensive above ~8 parameters. |

## Using with TMCMC

TMCMC also needs samples from the prior (not just density evaluations).
Use `prior.sample(n)`:

```python
prior2 = result1.as_prior(method='gaussian', discard=2000)

result2_tmcmc = mc.TMCMC(n_particles=1000).run(
    problem2,
    prior_samples=prior2.sample(1000),
)
```

## Initial covariance hint

`prior.cov` returns the sample covariance, which is a good starting value for
adaptive samplers:

```python
result2 = mc.DRAM(n_samples=20_000, initial_cov=prior2.cov).run(
    problem2, x0=prior2.mean
)
```

---

::: mcmckit.core.sequential.PosteriorPrior
