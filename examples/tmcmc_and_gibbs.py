"""TMCMC and Gibbs sampling examples on a 2D correlated Gaussian.

TMCMC:
- Does not require an initial proposal covariance or step size
- Provides a log-evidence (log marginal likelihood) estimate
- Requires drawing prior samples explicitly

Gibbs:
- Updates one parameter (or block) at a time
- Natural when parameters have different scales or are weakly coupled
- Each block gets its own proposal std
"""

import numpy as np
import mcmckit as mc

# -----------------------------------------------------------------------
# Problem
# -----------------------------------------------------------------------
true_mean = np.array([2.0, -1.0])
true_cov  = np.array([[1.0, 0.8],
                       [0.8, 1.0]])
true_prec = np.linalg.inv(true_cov)

# Prior: uniform on [-10, 10]^2 (improper but fine for demo)
PRIOR_LO = np.array([-10.0, -10.0])
PRIOR_HI = np.array([ 10.0,  10.0])


def log_prior(theta):
    if np.any(theta < PRIOR_LO) or np.any(theta > PRIOR_HI):
        return -np.inf
    return 0.0


def log_likelihood(theta):
    diff = theta - true_mean
    return -0.5 * diff @ true_prec @ diff


problem = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["x", "y"],
    bounds=[(-10.0, 10.0), (-10.0, 10.0)],
)

# -----------------------------------------------------------------------
# TMCMC
# -----------------------------------------------------------------------
N_PARTICLES = 1000

# Draw prior samples manually (uniform here)
prior_samples = np.random.uniform(PRIOR_LO, PRIOR_HI, size=(N_PARTICLES, 2))

tmcmc = mc.TMCMC(n_particles=N_PARTICLES, n_mcmc_steps=3)
result_tmcmc = tmcmc.run(problem, prior_samples=prior_samples)

print("TMCMC result:", result_tmcmc)
print("  mean       :", result_tmcmc.mean())
print("  std        :", result_tmcmc.std())
print(f"  log-evidence: {result_tmcmc.log_evidence:.4f}")
print(f"  stages     : {tmcmc.stage}")
print()

# Stage-by-stage execution (inspect beta progression)
print("--- Stage-by-stage TMCMC ---")
prior_samples2 = np.random.uniform(PRIOR_LO, PRIOR_HI, size=(N_PARTICLES, 2))
tmcmc2 = mc.TMCMC(n_particles=N_PARTICLES, n_mcmc_steps=3)
tmcmc2.initialize_with_samples(problem, prior_samples2)

while tmcmc2.beta < 1.0:
    tmcmc2.run_stage()
    print(f"  stage {tmcmc2.stage:2d} | beta={tmcmc2.beta:.4f} | log_ev={tmcmc2.log_evidence:.3f}")

result_tmcmc2 = tmcmc2.get_result()
print()

# -----------------------------------------------------------------------
# Gibbs
# -----------------------------------------------------------------------
# Scalar Gibbs (one param at a time, same proposal std)
gibbs_scalar = mc.Gibbs(n_samples=10_000, proposal_std=0.8)
result_gibbs = gibbs_scalar.run(problem, x0=[0.0, 0.0])

print("Gibbs (scalar) result:", result_gibbs)
print("  mean:", result_gibbs.mean())
print("  per-block acc:", gibbs_scalar.block_acceptance_rates)
print()

# Block Gibbs with different proposal stds per block
gibbs_block = mc.Gibbs(
    n_samples=10_000,
    blocks=[[0], [1]],       # still scalar but with per-param tuning
    proposal_std=[0.8, 0.8],
)
result_gibbs_block = gibbs_block.run(problem, x0=[0.0, 0.0])
print("Gibbs (per-block std) result:", result_gibbs_block)
print("  mean:", result_gibbs_block.mean())
print("  per-block acc:", gibbs_block.block_acceptance_rates)

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    result_tmcmc.plot_corner(true_values=true_mean,
                             title=f"TMCMC  log_evidence={result_tmcmc.log_evidence:.2f}")

    # Stage-by-stage particle evolution (light = prior, dark = posterior)
    tmcmc.plot_stages(title=f"TMCMC particle evolution  ({tmcmc.stage} stages)")

    result_gibbs.plot_corner(true_values=true_mean, title="Gibbs (scalar)")

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
