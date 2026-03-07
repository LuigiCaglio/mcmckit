# Changelog

All notable changes to mcmckit are documented here.

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
