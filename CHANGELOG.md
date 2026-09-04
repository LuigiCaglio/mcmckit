# Changelog

All notable changes to mcmckit are documented here.

## [Unreleased]

### Fixed
- **Three examples crashed on Windows.** `model_averaging.py`,
  `diagnostics_multichain.py` and `structural_identification.py` printed Greek
  letters and subscripts. A Windows console defaults to a legacy codepage
  (cp1252 in western Europe), not UTF-8, so printing a character outside it
  raises `UnicodeEncodeError` and kills the script. Printed text is now ASCII
  throughout; comments, docstrings and matplotlib labels keep their unicode,
  since the renderer handles it and it never reaches a console.

### Added
- CI now runs on **Windows and macOS**, not Linux alone, and covers Python
  3.13. The bug above shipped precisely because a Linux-only matrix could not
  see it, and the parallel module's documented Windows behaviour (workers
  re-import the calling module) was never exercised either.
- `tests/test_examples.py`: every example must parse, must print only ASCII,
  and must carry a `__main__` guard if it starts worker processes.

## [0.3.0] - 2026-09-04

### Added
- **A single-step API: `mcmckit.steps`.** Every sampler is now available as a
  plain function that advances a chain by one step, with all state threaded in
  and out explicitly. You write the recursion:

  ```python
  for i in range(1, n_iter + 1):
      x, logp, S, accepted = mc.ram_step(log_post, x, logp, S, i)
  ```

  `mh_step`, `ram_step`, `dram_step`, `mala_step`, `adaptive_mala_step` and
  `gibbs_step`. Nothing is stored on an object, so the loop, the storage, the
  stopping rule and the checkpointing are all the caller's. This is the
  interface the package now leads with; the sampler classes remain for
  stop-and-inspect workflows.
- **Full-run helpers: `mcmckit.runners`.** `metropolis`, `ram`, `dram`, `mala`,
  `adaptive_mala` and `gibbs` are thin loops over the step functions above, so
  both interfaces run identical code - verified by a test asserting that a
  helper and a hand-written loop produce identical chains from the same seed.
  They take either a bare log-posterior callable or a `Problem`.
- **Auxiliary model output passthrough.** If the log-posterior callable returns
  `(value, aux)`, the payload is threaded back out attached to the accepted
  sample and passed back in via `aux=`. Natural frequencies, mode shapes or
  residuals computed by the forward model are kept without re-running it,
  making a posterior predictive free. A callable returning a plain float gets
  no `aux` back, so the signature keeps its usual width.
- `examples/own_loop.py`, the new leading example: a 3-storey shear building
  identified from its natural frequencies in a hand-written RAM loop, with
  auxiliary output and an early stop. Documented at `docs/examples/own_loop.md`.
- 20 tests covering the new API, taking the suite from 104 to 124. Each step
  function is driven from a hand-written loop and must recover the mean and
  covariance of a correlated Gaussian; RAM must adapt away from a deliberately
  terrible initial scale; seeded runs must be reproducible; and the auxiliary
  payload must always correspond to the returned position.

### Changed
- **`Result.log_posteriors` is now optional**, so a chain you produced yourself
  wraps straight into `Result` for its plots: `mc.Result(samples=chain,
  param_names=[...])`.
- **RAM's default adaptation decay is now `gamma=0.51`**, down from 0.7, and is
  exposed as a keyword argument on both `ram_step` and `ram`.
- README, `docs/index.md` and `docs/quickstart.md` lead with the single-step
  interface, with the class interface documented after it.
- The install instructions now give the git URL. `pip install mcmckit` was
  documented but the package is not on PyPI.

### Packaging
- Distribution metadata for a PyPI release: `authors` and `maintainers` (the
  built wheel previously credited nobody, while `CITATION.cff` and
  `.zenodo.json` both named the author with an ORCID), `project.urls` pointing
  at the docs, repository, changelog and issues, keywords, and classifiers for
  Python 3.13 and the operating-system-independent pure-Python wheel. The
  summary now matches the project tagline.
- `[tool.setuptools.packages.find]` restricts the distribution to `mcmckit*`.
  `tests/` is an importable package, so flat-layout autodiscovery could
  otherwise have shipped it as a top-level `tests` module.
- `.github/workflows/publish.yml` builds, validates and smoke-tests the wheel,
  then publishes on a GitHub Release using PyPI Trusted Publishing, so no API
  token is stored in the repository or in GitHub secrets.

### Fixed
- **DRAM recomputed the empirical covariance over the entire chain history at
  every adaptation**, an O(n) cost per step and O(n^2) over a run. `dram_step`
  carries a running mean and sum of squared deviations (Welford) in its
  `DRAMState`, making adaptation O(d^2) per step regardless of chain length. A
  test asserts the recursive result equals `np.cov` over the full history.
- `site/`, the MkDocs build output, was not ignored by git.

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
  - `bayes_factor`: the arithmetic, the Jeffreys labels and their boundaries,
    overflow on enormous evidence ratios, and one end-to-end check where two
    conjugate models have analytic evidences so the Bayes factor is known.
  - `PosteriorPrior`: that it is a properly normalised density (checked against
    `scipy.stats` and by integrating it), and the property sequential updating
    exists for - updating on two batches in turn lands where updating on both
    at once does, against the conjugate closed form.
  - `HierarchicalProblem`: the parameter-vector layout round-trips, the joint
    density decomposes correctly, and sampling it reproduces the analytic
    population posterior and the analytic shrinkage factor
    `tau^2 / (tau^2 + s^2)`.

### Fixed
- **`Result.cov()` collapsed to a 0-d scalar for a single-parameter problem.**
  `mean()` and `std()` return shape `(n_params,)` at every size, but `np.cov`
  drops to a scalar in one dimension, so `result.cov()[0, 0]` raised an
  `IndexError` and the result could not be handed to a sampler's `initial_cov`.
  It is now always `(n_params, n_params)`.
- `HierarchicalProblem` documented that a single callable passed as
  `group_likelihoods` would be broadcast to all groups. The code rejects it. The
  docstring now matches the behaviour.

### Removed
- **`ModalLikelihood`.** It was structural model-updating machinery - forward
  models, mode pairing, per-mode noise - inside what is otherwise a
  general-purpose sampling library, and nothing else in the package imported it.
  Its `mac_matrix` helper survives as `mcmckit.mac` / `mcmckit.mac_matrix` in
  the new `core/similarity` module, which is domain-neutral: a squared
  normalised inner product, invariant to sign and scale. Use
  `GaussianNoiseLikelihood`, or a plain function, to build a modal likelihood on
  top of mcmckit rather than inside it.
  - The helper now handles **complex** vectors with the conjugate inner
    product. The old code cast to `float`, which discarded the imaginary part
    of a complex mode shape with only a warning and returned a wrong number.
  - `examples/modal_likelihood.py` and its documentation page were removed with
    it. `examples/structural_identification.py` still shows stiffness
    identification from modal data, built on the general API.

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
