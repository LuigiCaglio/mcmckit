# Changelog

All notable changes to mcmckit are documented here.

## [0.2.0] - 2026-09-03

### Added
- **Parallel evaluation** (`mcmckit.core.parallel`), opt-in via `n_workers` on
  `TMCMC` and `run_chains`. `n_workers=-1` uses one worker per core. TMCMC
  particles and independent chains are parallelised; a single chain is
  sequential by construction and is not.
  - `backend="process"` (default via `"auto"`), `backend="thread"` for
    likelihoods that release the GIL and are known to be thread-safe.
  - Process workers are pinned to one BLAS thread. NumPy already spreads a
    single operation over every core, so N workers otherwise request
    N x cores threads; that contention made 8 workers *slower* than serial in
    measurement.
  - Measured on 14 cores, 200 particles, single-threaded ~4 ms likelihood:
    2.60x on 4 workers, 3.36x on 8. `run_chains` with 4 chains: 2.72x.
  - TMCMC gives bit-identical results with and without workers; the random
    draws stay in the parent process and only likelihood calls are farmed out.
    `run_chains` with workers is not seed-reproducible, which is documented.
- `examples/openseespy_parallel.py`: model updating with a black-box OpenSeesPy
  forward model, verified to give bit-identical results at 1, 4 and 8 workers.
- Documentation page on parallel evaluation, including the rules for stateful
  black-box solvers.
- `__version__` on the package.
- 38 tests, taking the suite from 12 to 50:
  - Correctness rather than smoke: every sampler must recover the mean and
    covariance of a correlated Gaussian, and TMCMC must recover the analytic
    log-evidence of a conjugate model (it lands within ~0.03).
  - Gelman-Rubin must call converged chains converged and unmixed chains not;
    ESS must be bounded by the sample count.
  - Parallel results must match serial.

### Fixed
- **`TMCMC(n_workers=...)` was slower than running serially.** It built a fresh
  `ProcessPoolExecutor` on every call to `_eval_log_likelihoods` - once per
  stage plus `n_mcmc_steps` times per rejuvenation. On Windows each worker
  re-imports the calling module, so the pool churn dominated: 4 workers gave
  only 1.45x, and 8 workers were 1.6x slower than serial. One pool is now held
  open for the whole run.

### Changed
- **`backend="auto"` no longer falls back to threads for an unpicklable
  likelihood; it raises.** Threads are unsafe for a solver that keeps global
  state - OpenSeesPy holds a single global model domain, and threading it
  raised `OpenSeesError` on one run and segfaulted the interpreter on another.
  `auto` cannot know whether a likelihood is thread-safe, so it refuses rather
  than guessing. Threads remain available when requested explicitly.
- Picklability is checked before workers start, with a message naming the usual
  cause (a lambda or closure) and showing the fix, instead of a pickling
  traceback from inside the executor.
- `run_chains` never starts more workers than it has chains.

## [0.1.0] - 2025

### Added
- `Problem` class: black-box prior/likelihood interface with optional gradient support
- `Result` class: posterior sample container with summary statistics and rich visualisation
  - `plot_trace`, `plot_marginals`, `plot_corner` (4 styles: corner, scatter, full, kde)
  - `discard(n)` for burn-in removal; raw `samples` array access
- Samplers:
  - `MetropolisHastings`: random-walk MH with scalar/diagonal/full proposal covariance
  - `MALA`: gradient-biased proposals with asymmetric MH correction
  - `RAM`: Robust Adaptive Metropolis (Cholesky rank-1 update, Vihola 2012)
  - `DRAM`: Delayed Rejection Adaptive Metropolis (Haario et al. 2006)
  - `AdaptiveMALA`: MALA with log-space step-size adaptation
  - `TMCMC`: Transitional MCMC with adaptive β schedule, systematic resampling,
    log-evidence estimation, and `plot_stages()` for particle evolution visualisation
  - `Gibbs`: Metropolis-within-Gibbs with arbitrary block structure
- All samplers share `initialize / step / run / get_result` interface for step-by-step control
- Parallel likelihood evaluation via `ProcessPoolExecutor` (TMCMC)
