# Sequential Bayesian updating

In SHM and long-term monitoring, data arrives in batches over time — a new
measurement campaign after seasonal changes, after a reported event, or on a
scheduled interval.  Sequential Bayesian updating lets you incorporate each
new dataset without re-processing all historical data:

$$p(\theta \mid D_1, D_2) \propto p(D_2 \mid \theta)\; \underbrace{p(\theta \mid D_1)}_{\text{new prior}}$$

The posterior from step $t$ becomes the prior for step $t+1$.

## How it works in mcmckit

`PosteriorPrior` wraps a set of posterior samples and exposes a
`log_prior(theta)` callable — the same interface as any hand-written prior
function.  This means **all samplers work unchanged**.

```python
# After any sampling run
prior_next = result.as_prior(method='gaussian', discard=2000)

# Use exactly like a normal log_prior
problem_next = mc.Problem(prior=prior_next, likelihood=ll_new)
result_next  = mc.DRAM(...).run(problem_next, x0=prior_next.mean)
```

## Choosing the method

**`method='gaussian'`** — fits $\mathcal{N}(\mu, \Sigma)$ to the samples.

- Fast and memory-efficient in any dimension.
- `prior.mean` and `prior.cov` are immediately available.
- Best when the posterior is approximately unimodal and elliptical — the
  typical case for structural stiffness / damping identification.

**`method='kde'`** — non-parametric kernel density estimate.

- Handles multimodal or strongly skewed posteriors.
- Evaluation cost grows with the number of samples and dimension.
- Recommended for $\leq$ 6–8 parameters.

## Three-campaign example

```python
# Campaign 0: flat prior, 5 observations
result0 = mc.DRAM(n_samples=12_000, initial_cov=0.5*np.eye(2)).run(prob0, x0=[10., 8.])

# Campaign 1: Gaussian prior from campaign 0, 5 new observations
prior1 = result0.as_prior(method='gaussian', discard=2000)
prob1  = mc.Problem(prior=prior1, likelihood=ll1)
result1 = mc.DRAM(n_samples=12_000, initial_cov=prior1.cov).run(prob1, x0=prior1.mean)

# Campaign 2: further update, 10 new observations
prior2 = result1.as_prior(method='gaussian', discard=2000)
prob2  = mc.Problem(prior=prior2, likelihood=ll2)
result2 = mc.DRAM(n_samples=12_000, initial_cov=prior2.cov).run(prob2, x0=prior2.mean)
```

Posterior std across campaigns (2-DOF frame, k1=10, k2=8 N/m):

```
Campaign 0 (flat prior,     5 obs):  k1 std = 3.46,  k2 std = 1.72
Campaign 1 (Gaussian prior, +5 obs): k1 std = 2.35,  k2 std = 1.16
Campaign 2 (Gaussian prior,+10 obs): k1 std = 2.06,  k2 std = 1.03
```

Uncertainty shrinks monotonically as information accumulates.

## Use with TMCMC

TMCMC requires samples from the prior, not just density evaluations.
`PosteriorPrior.sample(n)` provides them:

```python
prior = result0.as_prior(method='gaussian', discard=2000)
result1 = mc.TMCMC(n_particles=500).run(
    problem1,
    prior_samples=prior.sample(500),
)
```

## Bayes consistency

Sequential updating is mathematically equivalent to a single batch run with
all data combined, provided the density approximation is accurate.  The
Gaussian method introduces a small approximation error when the posterior
deviates significantly from Gaussian; KDE reduces this at higher computational
cost.

## API reference

- [`PosteriorPrior`](../api/sequential.md)
- [`Result.as_prior()`](../api/result.md)

See the full runnable script at `examples/sequential.py`.
