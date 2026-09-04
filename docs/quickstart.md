# Quickstart

## Installation

```bash
pip install git+https://github.com/LuigiCaglio/mcmckit.git
```

---

# 1. You own the loop

This is the interface mcmckit is built around. Each sampler is a plain
function advancing the chain by **one step**, with all state threaded in and
out explicitly.

## Define a log posterior

One callable. Bigger means a better fit.

Every sampler is a plain module-level function: import what you use and call
it bare, or reach it through the namespace with `import mcmckit as mc`.

```python
import numpy as np
from mcmckit import ram_step

true_mean = np.array([2.0, -1.0])
true_prec = np.linalg.inv([[1.0, 0.8], [0.8, 1.0]])

def log_post(theta):
    if np.any(np.abs(theta) > 10):        # flat prior on [-10, 10]^2
        return -np.inf
    diff = theta - true_mean
    return -0.5 * diff @ true_prec @ diff
```

If your code carries a negative log-posterior, wrap it once:
`log_post = lambda th: -my_nll(th)`.

## Write the loop

```python
x = np.zeros(2)                              # current position
logp = log_post(x)                           # its log posterior
S = np.linalg.cholesky(np.eye(2) * 0.1**2)   # RAM adaptation state

chain = np.zeros((10_000, 2))
n_accepted = 0

for i in range(1, 10_001):
    x, logp, S, accepted = ram_step(log_post, x, logp, S, i)

    chain[i - 1] = x
    n_accepted += accepted

    if i % 1000 == 0:                        # your rules, your checkpoint
        print(f"step {i}: mean = {chain[:i].mean(0)}")

posterior = chain[2000:]                     # burn-in is a slice
print(posterior.mean(0), posterior.std(0))
print(f"acceptance: {n_accepted / 10_000:.2f}")
```

Stop early, save to disk, run your own convergence test, swap the step
function for another sampler. It is your loop.

## Keep your forward model's output

Return a tuple and the extra payload rides along with the accepted sample,
so a posterior predictive costs no extra model calls:

```python
def log_post(theta):
    freqs = my_forward_model(theta)
    return -0.5 * np.sum(((freqs - measured) / sigma)**2), freqs

x, logp, S, accepted, freqs = ram_step(log_post, x, logp, S, i, aux=freqs)
```

## Choosing a step function

| Function | Threaded state | Returns |
|---|---|---|
| `mh_step` | — (fixed `cov`) | `x, logp, accepted` |
| `ram_step` | `S`, step index `i` | `x, logp, S, accepted` |
| `dram_step` | `DRAMState` | `x, logp, state, accepted` |
| `mala_step` | `grad` | `x, logp, grad, accepted` |
| `adaptive_mala_step` | `grad`, `log_step`, `i` | `x, logp, grad, log_step, accepted` |
| `gibbs_step` | — (`blocks`, `proposal_std`) | `x, logp, accepted_per_block` |

`ram_step` is the best default: it self-tunes from a rough initial scale.
`dram_step` carries its state in a `DRAMState`, built by `init_dram_state(x0)`:

```python
from mcmckit import dram_step, init_dram_state

state = init_dram_state(x, initial_cov=0.1)
for i in range(n_iter):
    x, logp, state, accepted = dram_step(log_post, x, logp, state)
```

Gradient samplers take a callable returning `(log_post, grad)`:

```python
from mcmckit import mala_step

def log_post_and_grad(theta):
    diff = theta - true_mean
    return -0.5 * diff @ true_prec @ diff, -true_prec @ diff

logp, grad = log_post_and_grad(x)
for _ in range(n_iter):
    x, logp, grad, accepted = mala_step(log_post_and_grad, x, logp, grad, 0.4)
```

## Plot a chain you built yourself

```python
from mcmckit import Result

result = Result(samples=posterior, param_names=["x", "y"])
result.plot_corner(true_values=[2.0, -1.0], title="Posterior")
```

---

# 2. Or hand over the loop

The full-run helpers are thin loops over the same step functions.

```python
from mcmckit import ram

result = ram(log_post, x0=[0.0, 0.0], n_samples=10_000,
             param_names=["x", "y"])

result = result.discard(2000)
print(result.mean(), result.std(), result.acceptance_rate)

samples = result.samples          # raw array, shape (8000, 2)
```

`metropolis`, `ram`, `dram`, `mala`, `adaptive_mala` and `gibbs` all follow
this shape. Given the same seed they reproduce a hand-written loop exactly.

## Plot

```python
result.plot_corner(true_values=[2.0, -1.0], title="Posterior")
result.plot_trace()
result.plot_marginals()
```

Corner plot styles: `"corner"` (default), `"scatter"`, `"full"`, `"kde"`.

---

# 3. Problem objects and sampler classes

The class interface is still available, and takes a `Problem` bundling a
prior and a likelihood separately. Useful when you want a sampler you can
stop and inspect, and required for TMCMC.

```python
import mcmckit as mc

problem = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["x", "y"],
)

sampler = mc.DRAM(n_samples=10_000, initial_cov=np.eye(2) * 0.5)
result = sampler.run(problem, x0=[0.0, 0.0])
```

The full-run helpers accept a `Problem` too, in place of a bare callable.

## TMCMC

TMCMC advances a population of particles per stage rather than one chain
position, so it has no single-step form. It needs prior samples:

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

result = mc.dram(problem, x0=[...], n_samples=10_000)   # or: dram(problem, ...)

sigma_est = ll.posterior_sigma(result.mean())
```

See [Theory](theory.md) and [Noise estimation](examples/noise.md) for details.
