# mcmckit

Lightweight Bayesian model updating for engineering applications (structural dynamics, SHM, inverse problems).

## Install

```bash
pip install mcmckit           # core (numpy + scipy)
pip install mcmckit[plot]     # + matplotlib for plots
```

For development:

```bash
git clone https://github.com/your-username/mcmckit
cd mcmckit
pip install -e ".[dev,plot]"
```

## Quick start

```python
import numpy as np
import mcmckit as mc

# 1. Define the problem
def log_prior(theta):
    return 0.0  # flat prior

def log_likelihood(theta):
    return -0.5 * np.sum(theta**2)  # standard normal

problem = mc.Problem(prior=log_prior, likelihood=log_likelihood, param_names=["x", "y"])

# 2. Run a sampler
mh = mc.MetropolisHastings(proposal_cov=np.eye(2), n_samples=10_000)
result = mh.run(problem, x0=[0.0, 0.0])

# 3. Inspect results
print(result.mean(), result.std())
result.discard(1000).plot_corner(title="Posterior")
```

## Samplers

| Class | Method | Notes |
|---|---|---|
| `MetropolisHastings` | Random-walk MH | Fixed proposal covariance |
| `MALA` | Langevin | Requires `grad_log_likelihood` |
| `RAM` | Robust Adaptive MH | Self-tunes covariance (Vihola 2012) |
| `DRAM` | Delayed Rejection + AM | Best general-purpose adaptive sampler |
| `AdaptiveMALA` | Adaptive Langevin | Log-space step-size tuning |
| `TMCMC` | Transitional MCMC | Prior→posterior bridge, log-evidence estimate |
| `Gibbs` | Metropolis-within-Gibbs | Block updates, per-block acceptance rates |

All samplers share the same interface:

```python
sampler.initialize(problem, x0)   # set up state
sampler.step()                    # single step (stop any time)
result = sampler.get_result()     # collect samples so far

# or simply:
result = sampler.run(problem, x0)
```

## Problem definition

```python
problem = mc.Problem(
    prior=log_prior,                   # callable: theta -> float
    likelihood=log_likelihood,         # callable: theta -> float
    param_names=["E", "zeta"],         # optional
    bounds=[(-np.inf, np.inf), ...],   # optional, informational
    grad_log_likelihood=grad_ll,       # optional, required for MALA
    grad_log_prior=grad_lp,            # optional, required for MALA
)
```

## TMCMC (Transitional MCMC)

TMCMC bridges the prior to the posterior through tempered distributions
and provides a log-evidence (log marginal likelihood) estimate:

```python
prior_samples = np.random.uniform(-10, 10, size=(1000, 2))

tmcmc = mc.TMCMC(n_particles=1000, n_mcmc_steps=3)
result = tmcmc.run(problem, prior_samples=prior_samples)

print(f"log-evidence: {result.log_evidence:.3f}")

# Visualise particle evolution stage by stage
tmcmc.plot_stages(max_stages=6, title="Prior → Posterior")
```

## Visualisation

`Result` objects have built-in plotting:

```python
result.plot_trace()
result.plot_marginals()
result.plot_corner(style="corner", true_values=[2.0, -1.0])
# styles: "corner" | "scatter" | "full" | "kde"
```

## Examples

| Script | Demonstrates |
|---|---|
| `examples/simple_gaussian.py` | MH, all corner styles, burn-in, raw samples |
| `examples/mala_vs_mh.py` | MH vs MALA side-by-side |
| `examples/ram_example.py` | RAM self-tuning from bad initial covariance |
| `examples/adaptive_samplers.py` | MH / RAM / DRAM / AdaptiveMALA comparison |
| `examples/tmcmc_and_gibbs.py` | TMCMC stages + Gibbs scalar/block |

## Requirements

- Python ≥ 3.9
- numpy, scipy
- matplotlib (optional, for plots)
