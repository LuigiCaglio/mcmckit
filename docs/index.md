# mcmckit

**mcmckit** is a minimalistic Python package for Bayesian model updating, designed for engineering applications such as structural health monitoring (SHM), finite element model updating, and inverse problems.

---

## What it does

| Class | Method | Notes |
|---|---|---|
| `MetropolisHastings` | Random-walk MH | Fixed proposal covariance |
| `MALA` | Langevin | Gradient-biased proposals |
| `RAM` | Robust Adaptive MH | Self-tunes covariance (Vihola 2012) |
| `DRAM` | Delayed Rejection + AM | Best general-purpose adaptive sampler |
| `AdaptiveMALA` | Adaptive Langevin | Log-space step-size tuning |
| `TMCMC` | Transitional MCMC | Prior→posterior bridge, log-evidence estimate |
| `Gibbs` | Metropolis-within-Gibbs | Block updates, per-block acceptance rates |

All samplers share the same `initialize / step / run / get_result` interface, so you can stop at any point and inspect the chain.

---

## Installation

```bash
pip install mcmckit            # core (numpy + scipy)
pip install mcmckit[plot]      # + matplotlib for plots
pip install mcmckit[docs]      # + MkDocs for building these docs
```

Development install:

```bash
git clone https://github.com/your-username/mcmckit
cd mcmckit
pip install -e ".[dev,plot]"
```

---

## Quick example

```python
import numpy as np
import mcmckit as mc

def log_prior(theta):
    return 0.0   # flat prior

def log_likelihood(theta):
    return -0.5 * np.dot(theta, theta)   # standard normal

problem = mc.Problem(prior=log_prior, likelihood=log_likelihood,
                     param_names=["x", "y"])

result = mc.DRAM(n_samples=10_000, initial_cov=np.eye(2)).run(problem, x0=[0, 0])
result = result.discard(2000)   # burn-in

print(result.mean())            # → [~0, ~0]
result.plot_corner()
```

---

## Package layout

```
mcmckit/
├── core/
│   ├── problem.py           ← Problem (prior + likelihood interface)
│   ├── result.py            ← Result (samples, plots, statistics)
│   └── noise.py             ← GaussianNoiseLikelihood (fixed / estimated / marginalised)
└── samplers/
    ├── metropolis.py        ← MetropolisHastings
    ├── mala.py              ← MALA
    ├── ram.py               ← RAM
    ├── dram.py              ← DRAM
    ├── adaptive_mala.py     ← AdaptiveMALA
    ├── tmcmc.py             ← TMCMC
    └── gibbs.py             ← Gibbs
```
