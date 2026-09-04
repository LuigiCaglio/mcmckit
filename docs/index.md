# mcmckit

**mcmckit** is a minimal, plug-and-play set of MCMC samplers for Python, built
for Bayesian model updating and inverse problems in engineering: structural
health monitoring, finite element model updating, structural dynamics.

The expensive part of model updating is your forward model. mcmckit is
designed to stay out of its way.

---

## You own the loop

Every sampler is a plain function that advances the chain by **one step**.
State goes in as arguments and comes back as return values. Nothing is hidden
on an object, so the recursion is yours to write, stop, inspect and modify.

```python
import numpy as np
from mcmckit import ram_step

def log_post(theta):                          # your model goes here
    return -0.5 * np.sum(theta**2)

d = 2
x = np.zeros(d)                               # current position
logp = log_post(x)                            # its log posterior
S = np.linalg.cholesky(np.eye(d) * 0.1**2)    # RAM adaptation state

chain = np.zeros((10_000, d))
for i in range(1, 10_001):
    x, logp, S, accepted = ram_step(log_post, x, logp, S, i)

    chain[i - 1] = x
    if i % 1000 == 0:                         # your convergence check
        print(i, x, logp)
```

`ram_step` proposes, accepts or rejects, adapts the proposal covariance, and
hands everything back. Drop it into a loop you already have.

Everything is a plain module-level function, so import what you use and call
it bare, or keep the package namespace if you prefer. Both are the same call:

```python
from mcmckit import ram_step, dram_step      # bare
import mcmckit as mc                          # namespaced: mc.ram_step(...)
```

| Function | Threaded state | Returns |
|---|---|---|
| `mh_step` | — (fixed `cov`) | `x, logp, accepted` |
| `ram_step` | `S`, step index `i` | `x, logp, S, accepted` |
| `dram_step` | `DRAMState` | `x, logp, state, accepted` |
| `mala_step` | `grad` | `x, logp, grad, accepted` |
| `adaptive_mala_step` | `grad`, `log_step`, `i` | `x, logp, grad, log_step, accepted` |
| `gibbs_step` | — (`blocks`, `proposal_std`) | `x, logp, accepted_per_block` |

See [Step functions](api/steps.md) for the full reference, and
[Your own loop](examples/own_loop.md) for a worked structural example.

---

## Or hand over the loop

When you do not need control of the recursion, the full-run helpers are thin
loops over exactly the same step functions:

```python
from mcmckit import ram

result = ram(log_post, x0=[0.0, 0.0], n_samples=10_000)

print(result.mean(), result.std())
result.discard(1000).plot_corner()
```

`metropolis`, `ram`, `dram`, `mala`, `adaptive_mala` and `gibbs` all follow
this shape and return a [`Result`](api/result.md) with statistics and plots.

---

## Samplers

| Step function | Full run | Method | Notes |
|---|---|---|---|
| `mh_step` | `metropolis` | Random-walk MH | Fixed proposal covariance |
| `mala_step` | `mala` | Langevin | Needs the gradient |
| `ram_step` | `ram` | Robust Adaptive MH | Self-tunes covariance (Vihola 2012) |
| `dram_step` | `dram` | Delayed Rejection + AM | Best general-purpose adaptive sampler |
| `adaptive_mala_step` | `adaptive_mala` | Adaptive Langevin | Log-space step-size tuning |
| `gibbs_step` | `gibbs` | Metropolis-within-Gibbs | Block updates, per-block rates |
| — | `TMCMC` | Transitional MCMC | Prior→posterior bridge, log-evidence |

TMCMC advances a whole population of particles per stage rather than a single
chain position, so it has no single-step form and stays a class.

---

## Installation

```bash
pip install git+https://github.com/LuigiCaglio/mcmckit.git
```

With plotting, or for development:

```bash
pip install "mcmckit[plot] @ git+https://github.com/LuigiCaglio/mcmckit.git"

git clone https://github.com/LuigiCaglio/mcmckit
cd mcmckit
pip install -e ".[dev,plot]"
```

---

## Package layout

```
mcmckit/
├── steps.py                 ← single-step functions: you own the loop
├── runners.py               ← full-run helpers: thin loops over steps.py
├── core/
│   ├── problem.py           ← Problem (prior + likelihood interface)
│   ├── result.py            ← Result (samples, plots, statistics)
│   └── noise.py             ← GaussianNoiseLikelihood
└── samplers/                ← stateful sampler classes, and TMCMC
```
