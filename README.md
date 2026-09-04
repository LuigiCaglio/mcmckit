# mcmckit

Minimal, plug-and-play MCMC samplers for Python.

Built for Bayesian model updating and inverse problems in engineering, where
the expensive part is your forward model and you want the sampler to stay out
of the way. Depends only on NumPy and SciPy.

## Install

```bash
pip install git+https://github.com/LuigiCaglio/mcmckit.git
```

For development:

```bash
git clone https://github.com/LuigiCaglio/mcmckit
cd mcmckit
pip install -e ".[dev,plot]"
```

## Quick start — you own the loop

Every sampler is a plain function that advances the chain by **one step**.
State goes in as arguments and comes back as return values. Nothing is hidden
on an object, so the recursion is yours to write, stop, inspect and modify:

```python
import numpy as np
import mcmckit as mc

def log_post(theta):                      # your model goes here
    return -0.5 * np.sum(theta**2)

d = 2
x = np.zeros(d)                           # current position
logp = log_post(x)                        # its log posterior
S = np.linalg.cholesky(np.eye(d) * 0.1**2)   # RAM adaptation state

chain = np.zeros((10_000, d))
for i in range(1, 10_001):
    x, logp, S, accepted = mc.ram_step(log_post, x, logp, S, i)

    chain[i - 1] = x
    if i % 1000 == 0:                     # your convergence check, your rules
        print(i, x, logp)
```

That is the whole interface. `ram_step` proposes, accepts or rejects, adapts
the proposal covariance, and hands everything back. Drop it into a loop you
already have.

| Function | Threaded state | Returns |
|---|---|---|
| `mh_step` | — (fixed `cov`) | `x, logp, accepted` |
| `ram_step` | `S`, step index `i` | `x, logp, S, accepted` |
| `dram_step` | `DRAMState` | `x, logp, state, accepted` |
| `mala_step` | `grad` | `x, logp, grad, accepted` |
| `adaptive_mala_step` | `grad`, `log_step`, `i` | `x, logp, grad, log_step, accepted` |
| `gibbs_step` | — (`blocks`, `proposal_std`) | `x, logp, accepted_per_block` |

### Sign convention

Every function takes a **log-posterior**: bigger is a better fit. If your code
carries a negative log-posterior, wrap it once:

```python
log_post = lambda theta: -my_nll(theta)
```

### Keeping your forward model's output

If your callable returns `(log_post, aux)`, the extra payload rides along and
comes back attached to the accepted sample. Natural frequencies, mode shapes,
residuals — whatever your model already computed, without re-running it:

```python
def log_post(theta):
    freqs = surrogate(theta)
    return -0.5 * np.sum(((freqs - measured) / sigma)**2), freqs

x, logp, S, accepted, freqs = mc.ram_step(log_post, x, logp, S, i, aux=freqs)
```

Return a plain float instead and no `aux` comes back, so the signature stays
its usual width.

## Or hand over the loop

When you do not need control of the recursion, the full-run helpers are thin
loops over exactly the same step functions:

```python
result = mc.ram(log_post, x0=[0.0, 0.0], n_samples=10_000)

print(result.mean(), result.std())
result.discard(1000).plot_corner(title="Posterior")
```

`metropolis`, `ram`, `dram`, `mala`, `adaptive_mala` and `gibbs` all follow
this shape and return a `Result` with statistics and plots attached.

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

TMCMC works on a population of particles per stage rather than one chain
position, so it does not have a single-step form. It stays a class.

### Stateful sampler objects

The original class interface is still there for stop-and-inspect workflows,
and is unchanged:

```python
sampler = mc.RAM(n_samples=10_000, initial_cov=np.eye(2))
sampler.initialize(problem, x0=[0.0, 0.0])
sampler.step()
result = sampler.get_result()
```

These take a `Problem`, which bundles a prior and likelihood:

```python
problem = mc.Problem(prior=log_prior, likelihood=log_likelihood,
                     param_names=["E", "zeta"])
```

The full-run helpers accept either a `Problem` or a bare `log_post` callable.

## TMCMC (Transitional MCMC)

TMCMC bridges the prior to the posterior through tempered distributions
and provides a log-evidence (log marginal likelihood) estimate:

```python
prior_samples = np.random.uniform(-10, 10, size=(1000, 2))

tmcmc = mc.TMCMC(n_particles=1000, n_mcmc_steps=3)
result = tmcmc.run(problem, prior_samples=prior_samples)

print(f"log-evidence: {result.log_evidence:.3f}")
tmcmc.plot_stages(max_stages=6, title="Prior → Posterior")
```

## Parallel evaluation

Opt-in, off by default. The cost in model updating is the forward model inside
your likelihood, so mcmckit can spread those calls over cores:

```python
if __name__ == "__main__":                      # required: workers re-import the module
    result = mc.TMCMC(n_particles=1000, n_workers=4).run(problem, prior_samples=ps)
    multi = mc.run_chains(sampler, problem, x0, n_chains=4, n_workers=4)
```

`n_workers=-1` uses one worker per core. TMCMC particles and independent chains
are parallelised; a single chain is sequential by construction and is not.

Process workers pickle your likelihood, so it must be a module-level function
rather than a lambda or closure; mcmckit says so clearly instead of failing deep
inside the executor. TMCMC gives bit-identical results with and without workers.

On a 14-core machine with a ~4 ms likelihood: 2.60x on 4 workers, 3.36x on 8.

Black-box solvers work — `examples/openseespy_parallel.py` updates a 6-storey
shear building through OpenSeesPy and gets bit-identical results at 1, 4 and 8
workers. Use **processes, not threads**, for any solver with global state:
OpenSeesPy keeps one global model domain, and threading it raises `OpenSeesError`
or segfaults. See [the docs](docs/parallel.md).

## Visualisation

`Result` objects have built-in plotting:

```python
result.plot_trace()
result.plot_marginals()
result.plot_corner(style="corner", true_values=[2.0, -1.0])
# styles: "corner" | "scatter" | "full" | "kde"
```

Driving the loop yourself gives you a plain array, which you can wrap when you
want those plots:

```python
result = mc.Result(samples=chain, param_names=["E", "zeta"])
result.plot_corner()
```

## Examples

| Script | Demonstrates |
|---|---|
| `examples/own_loop.py` | **Start here.** Step functions, your own loop, early stopping, aux output |
| `examples/simple_gaussian.py` | MH, all corner styles, burn-in, raw samples |
| `examples/mala_vs_mh.py` | MH vs MALA side-by-side |
| `examples/ram_example.py` | RAM self-tuning from bad initial covariance |
| `examples/adaptive_samplers.py` | MH / RAM / DRAM / AdaptiveMALA comparison |
| `examples/tmcmc_and_gibbs.py` | TMCMC stages + Gibbs scalar/block |
| `examples/diagnostics_multichain.py` | Multi-chain runs, R-hat and ESS diagnostics |
| `examples/noise_estimation.py` | Inferring measurement noise alongside parameters |
| `examples/structural_identification.py` | Stiffness identification from modal data |
| `examples/hierarchical_updating.py` | Hierarchical models across multiple structures |
| `examples/model_comparison.py` | Bayes factors from TMCMC log-evidence |
| `examples/model_averaging.py` | Posterior model averaging |
| `examples/sequential.py` | Sequential / online updating with `PosteriorPrior` |
| `examples/sequential_10dof.py` | 10-parameter sequential structural identification |
| `examples/openseespy_parallel.py` | Black-box OpenSeesPy forward model, run in parallel |

## Citing

If you use mcmckit in academic work, please cite it. See `CITATION.cff`, or use
the DOI minted for the release you used.

## Requirements

- Python ≥ 3.9
- numpy, scipy
- matplotlib (optional, for plots)
