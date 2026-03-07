"""Convergence diagnostics and multi-chain MCMC.

Demonstrates how to:
  1. Run multiple independent chains with run_chains()
  2. Check convergence via the Gelman-Rubin R-hat summary table
  3. Inspect per-parameter autocorrelation and ESS on a single chain
  4. Pool chains after burn-in for final inference

Scenario: identify two stiffness values of a 2-DOF shear frame from
noisy natural frequency measurements.  This is the same forward model used
in the structural identification example, kept deliberately simple so the
focus stays on the diagnostics.

True system:  m1 = m2 = 1 kg,  k1 = 10 N/m,  k2 = 8 N/m
Observations: 8 measurements of each natural frequency, sigma_obs = 0.05 rad/s
"""

import numpy as np
import mcmckit as mc
from scipy.linalg import eigh

rng = np.random.default_rng(42)

# -----------------------------------------------------------------------
# System and simulated data
# -----------------------------------------------------------------------
TRUE_K1 = 10.0
TRUE_K2 = 8.0
SIGMA_OBS = 0.05
N_OBS = 8


def natural_frequencies(k1, k2, m1=1.0, m2=1.0):
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    M = np.diag([m1, m2])
    vals, _ = eigh(K, M)
    return np.sqrt(np.maximum(vals, 0.0))


true_freqs = natural_frequencies(TRUE_K1, TRUE_K2)
y_obs = np.concatenate([
    rng.normal(true_freqs[0], SIGMA_OBS, size=N_OBS),
    rng.normal(true_freqs[1], SIGMA_OBS, size=N_OBS),
])

print(f"True frequencies: ω1={true_freqs[0]:.4f}, ω2={true_freqs[1]:.4f} rad/s")
print(f"True stiffness:   k1={TRUE_K1}, k2={TRUE_K2}")
print()

# -----------------------------------------------------------------------
# Problem definition
# -----------------------------------------------------------------------
def log_prior(theta):
    k1, k2 = theta
    if k1 <= 0 or k2 <= 0:
        return -np.inf
    return (-0.5 * ((np.log(k1) - np.log(10)) / 1.0)**2
            - 0.5 * ((np.log(k2) - np.log(10)) / 1.0)**2)


likelihood = mc.GaussianNoiseLikelihood(
    forward_model=lambda theta: np.tile(natural_frequencies(theta[0], theta[1]), N_OBS),
    y_obs=y_obs,
    noise_std=SIGMA_OBS,
)

problem = mc.Problem(prior=log_prior, likelihood=likelihood,
                     param_names=["k1", "k2"])

# -----------------------------------------------------------------------
# Part 1: multi-chain DRAM
# -----------------------------------------------------------------------
print("=== Part 1: Multi-chain DRAM ===")

sampler = mc.DRAM(n_samples=12_000, initial_cov=0.5 * np.eye(2))

mc_result = mc.run_chains(
    sampler,
    problem,
    x0=[10.0, 8.0],   # single x0 — chains are jittered automatically
    n_chains=4,
)

print(mc_result)
print()

# Convergence summary (with 3000 burn-in)
mc_result.summary(discard=3000)
print()

# -----------------------------------------------------------------------
# Part 2: pool chains and compute final posterior
# -----------------------------------------------------------------------
print("=== Part 2: Pooled posterior ===")

pooled = mc_result.pool(discard=3000)
print(f"Pooled samples: {len(pooled.samples)}")
print(f"  k1: {pooled.mean()[0]:.3f} ± {pooled.std()[0]:.3f}  (true: {TRUE_K1})")
print(f"  k2: {pooled.mean()[1]:.3f} ± {pooled.std()[1]:.3f}  (true: {TRUE_K2})")
print()

# ESS on the pooled chain
pooled_ess = pooled.ess()
print(f"Pooled ESS:  k1={pooled_ess[0]:.0f},  k2={pooled_ess[1]:.0f}")
print()

# -----------------------------------------------------------------------
# Part 3: single-chain diagnostics
# -----------------------------------------------------------------------
print("=== Part 3: Single-chain diagnostics (chain 0) ===")

chain0 = mc_result[0].discard(3000)
chain0_ess = chain0.ess()
print(f"Chain 0 acceptance rate: {mc_result[0].acceptance_rate:.3f}"
      if mc_result[0].acceptance_rate is not None else "Chain 0")
print(f"Chain 0 ESS:  k1={chain0_ess[0]:.0f},  k2={chain0_ess[1]:.0f}")
print()

# Standalone R-hat (all 4 chains, after burn-in)
rhat = mc.gelman_rubin([r.discard(3000) for r in mc_result])
print(f"Gelman-Rubin R-hat:  k1={rhat[0]:.4f},  k2={rhat[1]:.4f}")
print()

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    # Trace plots for all 4 chains
    fig1 = mc_result.plot_traces(title="Trace plots — 4 chains (DRAM)")

    # Autocorrelation for chain 0 (after burn-in)
    fig2 = chain0.plot_autocorr(max_lag=80, title="Autocorrelation — chain 0")

    # Corner plot of the pooled posterior
    fig3 = pooled.plot_corner(
        true_values=[TRUE_K1, TRUE_K2],
        title="Pooled posterior (4 chains × 9000 samples)",
    )

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
