"""Bayesian Model Averaging (BMA) for structural prediction.

Scenario
--------
A 2-DOF shear frame is monitored.  We observe two natural frequencies but
we are uncertain about the boundary condition at the top floor:

  M1 — free top (cantilever-like):  k1 = k, k2 = k    (one stiffness)
  M2 — fixed top (symmetric):       k1, k2 independent  (two stiffnesses)

Both models are physically plausible a priori.  The data will update their
relative credibility via the evidence, and BMA will blend their predictions
proportionally to that credibility.

This example shows:
  1. Model weights and how they depend on the data
  2. BMA prediction vs single-best-model prediction
  3. decompose() — per-model contribution to the BMA
  4. Effect of informative prior model weights

True system: m1 = m2 = 1 kg,  k1 = 10 N/m,  k2 = 8 N/m  (M2 is correct)
"""

import numpy as np
import mcmckit as mc
from scipy.linalg import eigh

rng = np.random.default_rng(1)

# -----------------------------------------------------------------------
# True system
# -----------------------------------------------------------------------
TRUE_K1 = 10.0
TRUE_K2 = 8.0
SIGMA_OBS = 0.05
N_OBS = 6


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
print()

# -----------------------------------------------------------------------
# Model M1: single stiffness k (k1 = k2 = k)
# -----------------------------------------------------------------------
def fwd_m1(theta):
    k = max(float(theta[0]), 1e-6)
    return np.tile(natural_frequencies(k, k), N_OBS)


def log_prior_m1(theta):
    k = float(theta[0])
    if k <= 0:
        return -np.inf
    return -0.5 * ((np.log(k) - np.log(9)) / 1.0)**2


lik_m1 = mc.GaussianNoiseLikelihood(fwd_m1, y_obs, noise_std=SIGMA_OBS)
problem_m1 = mc.Problem(prior=log_prior_m1, likelihood=lik_m1,
                         param_names=["k"])

prior_m1 = rng.lognormal(np.log(9), 1.0, size=(1000, 1))

# -----------------------------------------------------------------------
# Model M2: two independent stiffnesses k1, k2
# -----------------------------------------------------------------------
def fwd_m2(theta):
    k1, k2 = float(theta[0]), float(theta[1])
    return np.tile(natural_frequencies(k1, k2), N_OBS)


def log_prior_m2(theta):
    k1, k2 = float(theta[0]), float(theta[1])
    if k1 <= 0 or k2 <= 0:
        return -np.inf
    return (-0.5 * ((np.log(k1) - np.log(9)) / 1.0)**2
            - 0.5 * ((np.log(k2) - np.log(9)) / 1.0)**2)


lik_m2 = mc.GaussianNoiseLikelihood(fwd_m2, y_obs, noise_std=SIGMA_OBS)
problem_m2 = mc.Problem(prior=log_prior_m2, likelihood=lik_m2,
                         param_names=["k1", "k2"])

prior_m2 = rng.lognormal(np.log(9), 1.0, size=(1000, 2))

# -----------------------------------------------------------------------
# Run model comparison + BMA
# -----------------------------------------------------------------------
comp = mc.ModelComparison(
    models=[
        ("M1: single k",   problem_m1, prior_m1),
        ("M2: k1 + k2",    problem_m2, prior_m2),
    ],
    tmcmc_kwargs={"n_particles": 1000, "n_mcmc_steps": 3},
)
comp.run()

print("=== Model comparison ===")
comp.summary()
print()

w = comp.weights()
print(f"Posterior model weights:")
for r, wi in zip(comp.summary(print_table=False), w):
    print(f"  {r['name']:<20}  w = {wi:.4f}")
print()

# -----------------------------------------------------------------------
# BMA prediction of natural frequencies
# -----------------------------------------------------------------------
bma = comp.predict(
    forward_models={
        "M1: single k":  lambda theta: natural_frequencies(theta[0], theta[0]),
        "M2: k1 + k2":   lambda theta: natural_frequencies(theta[0], theta[1]),
    },
    n_eval=2000,
)

print(f"BMA result: {bma}")
print()
print("BMA posterior predictive (natural frequencies):")
print(f"  ω1: {bma.mean()[0]:.4f} ± {bma.std()[0]:.4f}  (true: {true_freqs[0]:.4f})")
print(f"  ω2: {bma.mean()[1]:.4f} ± {bma.std()[1]:.4f}  (true: {true_freqs[1]:.4f})")
print()

# Per-model decomposition
print("Per-model contribution:")
dec = bma.decompose()
for name, info in dec.items():
    print(f"  {name:<20}  w={info['weight']:.4f}"
          f"  ω1_mean={info['mean'][0]:.4f}"
          f"  ω2_mean={info['mean'][1]:.4f}")
print()

# -----------------------------------------------------------------------
# Effect of informative prior model weights
# -----------------------------------------------------------------------
print("=== Informative prior: favour M1 (p(M1)=0.8, p(M2)=0.2) ===")
# Note: weights() order matches summary() — best model first
# After run(), results are sorted best→worst; check which is first
sorted_names = [r["name"] for r in comp.summary(print_table=False)]
# Build prior_weights matching that order
pw_dict = {"M1: single k": 0.8, "M2: k1 + k2": 0.2}
pw = np.array([pw_dict[n] for n in sorted_names])
w_informative = comp.weights(prior_weights=pw)
for name, wi in zip(sorted_names, w_informative):
    print(f"  {name:<20}  w = {wi:.4f}")
print()

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    obs_x = np.array([0, 1])   # two frequency indices
    obs_y = np.array([y_obs[:N_OBS].mean(), y_obs[N_OBS:].mean()])

    fig1 = bma.plot_bands(
        x=obs_x,
        y_obs=obs_y,
        xlabel="Frequency index",
        ylabel="ω (rad/s)",
        title="BMA posterior predictive (90% CI)",
    )
    # Add true values
    ax = fig1.axes[0]
    ax.scatter(obs_x, true_freqs, marker="*", s=120, color="gold",
               zorder=6, label="true")
    ax.legend(fontsize=8)

    fig2 = bma.plot_decompose(
        x=obs_x,
        xlabel="Frequency index",
        ylabel="ω (rad/s)",
        title="Per-model mean predictions (line width ∝ weight)",
    )

    # Evidence bar chart
    fig3 = comp.plot(title="Model comparison")

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
