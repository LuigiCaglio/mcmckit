# Parallel evaluation

In Bayesian model updating the cost is almost always the forward model inside
your log-likelihood — a finite element run, a time integration, a modal solve.
mcmckit can spread those calls across cores.

It is **opt-in and off by default**. Everything below changes how the work is
scheduled, never what is computed.

## Quick version

```python
# TMCMC: evaluate all particles in a stage at once
result = mc.TMCMC(n_particles=1000, n_workers=4).run(problem, prior_samples=ps)

# Independent chains, run at the same time
multi = mc.run_chains(sampler, problem, x0, n_chains=4, n_workers=4)
```

`n_workers=-1` uses one worker per core.

!!! warning "Put your script behind a `__main__` guard"
    Process workers start by importing the module that launched them. Without
    the guard your script re-runs itself in every worker.

    ```python
    if __name__ == "__main__":
        result = mc.TMCMC(n_particles=1000, n_workers=4).run(problem, prior_samples=ps)
    ```

    This matters on Windows and macOS, which spawn rather than fork. It costs
    nothing on Linux, so write it anyway.

## What is parallelised

| | Parallel | Why |
|---|---|---|
| TMCMC particles | yes | Every particle in a stage is independent |
| Independent chains | yes | Chains never talk to each other |
| A single chain | **no** | Each step depends on the last — sequential by construction |

If you are running one long chain, workers will not help. Run several shorter
chains with `run_chains` instead, which also gives you an R-hat.

## Your likelihood must be picklable

Process workers receive your likelihood by pickling it. That rules out lambdas,
closures, and functions defined inside another function:

```python
# will not pickle
problem = mc.Problem(prior=lambda t: 0.0, likelihood=lambda t: -0.5 * t @ t)

# will
def log_prior(theta): ...
def log_likelihood(theta): ...
problem = mc.Problem(prior=log_prior, likelihood=log_likelihood)
```

mcmckit checks this before starting, so you get a clear message naming the likely
cause rather than a pickling traceback from inside the executor. It refuses to
run instead of silently switching to threads, which would be unsafe for a solver
that keeps global state.

## Backends

```python
mc.TMCMC(n_particles=1000, n_workers=4, backend="process")
```

`"process"`
:   Separate processes. The right choice when your likelihood is pure Python,
    because the GIL otherwise serialises it. Needs a picklable likelihood.

`"thread"`
:   Threads. No pickling, no start-up cost, but it helps only when the
    likelihood releases the GIL, and it is **unsafe for any solver that keeps
    global state**. See the OpenSeesPy section below.

`"auto"` (default)
:   Processes when the likelihood is picklable. When it is not, `auto` **raises**
    rather than quietly switching to threads — it cannot know whether your
    likelihood is thread-safe, and guessing wrong corrupts results silently.

## Expected speedup

Measured on a 14-core machine, TMCMC with 200 particles and a single-threaded
~4 ms likelihood:

| Workers | Time | Speedup |
|---:|---:|---:|
| 1 | 6.15 s | 1.00x |
| 2 | 3.63 s | 1.69x |
| 4 | 2.36 s | 2.60x |
| 8 | 1.83 s | 3.36x |

`run_chains` with 4 chains: 7.32 s to 2.69 s, a 2.72x speedup.

Scaling is sublinear because each stage has a serial section — reweighting,
resampling, the covariance estimate — and because dispatching work to processes
costs something per batch. The gain grows with the cost of your likelihood. If a
single evaluation takes milliseconds, the overhead may dominate; if it takes a
second, expect close to linear scaling.

!!! note "Thread oversubscription"
    If your likelihood calls into NumPy, BLAS is *already* using every core for
    a single evaluation. Running N such workers then asks for N x cores threads,
    and the contention can make the parallel run slower than the serial one.
    mcmckit pins each process worker to one BLAS thread to prevent exactly
    that. If you have your own reason to override it, construct the pool
    directly with `WorkerPool(..., limit_blas=False)`.

## Reproducibility

The serial path is reproducible under `np.random.seed`. TMCMC stays reproducible
with workers too, because the random draws happen in the parent process and only
the likelihood evaluations are farmed out — `test_parallel.py` asserts that
serial and parallel runs agree exactly.

`run_chains` with `n_workers > 1` is **not** reproducible that way: each worker
process seeds its own stream. Use `n_workers=1` when you need bit-for-bit
repeatability.

## Black-box solvers: OpenSeesPy and friends

This is the normal case in structural model updating — the likelihood builds and
solves a finite element model, and mcmckit only ever calls it. It works, with one
rule: **use processes, never threads.**

OpenSeesPy keeps a *single global model domain*. `ops.model(...)` in two threads
at once means both are writing the same model. Measured on a 6-storey shear
building, the thread backend raised `OpenSeesError` on one run and segfaulted the
interpreter on another — a nondeterministic failure, the worst kind to debug.
Separate processes each get their own domain, so they cannot interfere.

With `backend="process"` the same model updating gave **bit-identical** posterior
means and log-evidence at 1, 4 and 8 workers. `examples/openseespy_parallel.py`
is the full runnable case.

The pattern:

```python
import numpy as np
import openseespy.opensees as ops
import mcmckit as mc

F_OBS = ...                      # module level: every worker gets it on import

def build_and_eigen(scale):
    ops.wipe()                   # tear the global domain down every call
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ...
    frequencies = ops.eigen("-fullGenLapack", n_modes)
    ops.wipe()
    return frequencies

def log_likelihood(theta):       # module level, so it pickles
    return -0.5 * np.sum(((build_and_eigen(theta) - F_OBS) / sigma) ** 2)

PROBLEM = mc.Problem(prior=log_prior, likelihood=log_likelihood)

if __name__ == "__main__":       # required
    result = mc.TMCMC(n_particles=400, n_workers=4, backend="process").run(
        PROBLEM, prior_samples=ps
    )
```

Four things to get right:

1. **`ops.wipe()` at both ends of the forward model.** The domain is global and
   persists between calls, so a model that is not torn down leaks into the next
   evaluation.
2. **Module-level likelihood**, so it can be pickled to the workers.
3. **Module-level data** (`F_OBS` above). Each worker imports the module, so
   anything defined there is available without being sent per call.
4. **The `__main__` guard**, or every worker re-runs your script.

If you already parallelise OpenSeesPy with `joblib.Parallel`, this is the same
mechanism — joblib's default `loky` backend is process-based for exactly this
reason.

!!! tip "Is your model big enough to be worth it?"
    Time a single likelihood call first. The example model solves in ~2 ms and
    is *slower* with 4 workers, because dispatching costs more than the solve.
    Workers start paying once one evaluation takes tens of milliseconds — a
    nonlinear time history rather than a handful of eigenvalues.
