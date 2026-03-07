"""Bayesian model class selection for structural identification.

Scenario
--------
A 2-DOF shear frame (two storeys) is monitored under ambient vibration.  We
observe its two natural frequencies with Gaussian noise.  Three model classes
compete:

  M1 — 1-DOF (one stiffness k, mass m fixed):       wrong model, too simple
  M2 — 2-DOF (two stiffnesses k1, k2, masses fixed): correct model
  M3 — 2-DOF + free mass ratio r = m2/m1:            over-parameterised

The marginal likelihood (evidence) p(y|M) is estimated by TMCMC and used to
compute Bayes factors on the Jeffreys scale.

True system:  m1 = m2 = 1 kg,  k1 = 10 N/m,  k2 = 8 N/m
Observations: two natural frequencies with sigma_obs = 0.05 rad/s
"""

import numpy as np
import mcmckit as mc

rng = np.random.default_rng(0)

# -----------------------------------------------------------------------
# True system and simulated data
# -----------------------------------------------------------------------
M1_TRUE = 1.0
M2_TRUE = 1.0
K1_TRUE = 10.0
K2_TRUE = 8.0
SIGMA_OBS = 0.05


def natural_frequencies_2dof(k1, k2, m1=1.0, m2=1.0):
    """Exact natural frequencies of a 2-DOF shear frame (rad/s)."""
    K = np.array([[k1 + k2, -k2],
                  [-k2,      k2]])
    M = np.diag([m1, m2])
    # Solve generalised eigenvalue problem K v = omega^2 M v
    from scipy.linalg import eigh
    vals, _ = eigh(K, M)
    return np.sqrt(np.maximum(vals, 0))   # shape (2,)


true_freqs = natural_frequencies_2dof(K1_TRUE, K2_TRUE)
print(f"True natural frequencies: {true_freqs[0]:.4f}, {true_freqs[1]:.4f} rad/s")

y_obs = rng.normal(true_freqs, SIGMA_OBS)
print(f"Observed frequencies:      {y_obs[0]:.4f}, {y_obs[1]:.4f} rad/s")
print()

# -----------------------------------------------------------------------
# Model M1 — 1-DOF  (param: [k])
# -----------------------------------------------------------------------
#   Single stiffness, single mass m=1.  Forward model: omega = sqrt(k/m).
#   We fit one equivalent frequency (average of two observed).
#   This is genuinely the wrong model — it cannot reproduce two independent
#   frequencies with one parameter.

def fwd_m1(theta):
    k = float(theta[0])
    omega = np.sqrt(max(k, 1e-6))
    return np.array([omega, omega])   # same frequency for both observations


def log_prior_m1(theta):
    k = float(theta[0])
    # Lognormal-ish: Normal on log(k) centred on log(10)
    if k <= 0:
        return -np.inf
    return -0.5 * ((np.log(k) - np.log(10)) / 1.0)**2


lik_m1 = mc.GaussianNoiseLikelihood(fwd_m1, y_obs, noise_std=SIGMA_OBS)
problem_m1 = mc.Problem(prior=log_prior_m1, likelihood=lik_m1,
                         param_names=["k"])

prior_m1 = np.column_stack([
    rng.lognormal(np.log(10), 1.0, size=1000),
])

# -----------------------------------------------------------------------
# Model M2 — 2-DOF  (params: [k1, k2])
# -----------------------------------------------------------------------

def fwd_m2(theta):
    k1, k2 = float(theta[0]), float(theta[1])
    return natural_frequencies_2dof(k1, k2)


def log_prior_m2(theta):
    k1, k2 = float(theta[0]), float(theta[1])
    if k1 <= 0 or k2 <= 0:
        return -np.inf
    lp  = -0.5 * ((np.log(k1) - np.log(10)) / 1.0)**2
    lp += -0.5 * ((np.log(k2) - np.log(10)) / 1.0)**2
    return lp


lik_m2 = mc.GaussianNoiseLikelihood(fwd_m2, y_obs, noise_std=SIGMA_OBS)
problem_m2 = mc.Problem(prior=log_prior_m2, likelihood=lik_m2,
                         param_names=["k1", "k2"])

prior_m2 = np.column_stack([
    rng.lognormal(np.log(10), 1.0, size=1000),
    rng.lognormal(np.log(10), 1.0, size=1000),
])

# -----------------------------------------------------------------------
# Model M3 — 2-DOF + free mass ratio  (params: [k1, k2, r])
#   r = m2/m1,  m1 fixed to 1.  Extra free parameter → Occam penalty.
# -----------------------------------------------------------------------

def fwd_m3(theta):
    k1, k2, r = float(theta[0]), float(theta[1]), float(theta[2])
    m2 = max(r, 1e-3)
    return natural_frequencies_2dof(k1, k2, m1=1.0, m2=m2)


def log_prior_m3(theta):
    k1, k2, r = float(theta[0]), float(theta[1]), float(theta[2])
    if k1 <= 0 or k2 <= 0 or r <= 0:
        return -np.inf
    lp  = -0.5 * ((np.log(k1) - np.log(10)) / 1.0)**2
    lp += -0.5 * ((np.log(k2) - np.log(10)) / 1.0)**2
    lp += -0.5 * ((np.log(r)  - 0.0)        / 1.0)**2   # lognormal centred on 1
    return lp


lik_m3 = mc.GaussianNoiseLikelihood(fwd_m3, y_obs, noise_std=SIGMA_OBS)
problem_m3 = mc.Problem(prior=log_prior_m3, likelihood=lik_m3,
                         param_names=["k1", "k2", "r"])

prior_m3 = np.column_stack([
    rng.lognormal(np.log(10), 1.0, size=1000),
    rng.lognormal(np.log(10), 1.0, size=1000),
    rng.lognormal(0.0, 1.0,        size=1000),
])

# -----------------------------------------------------------------------
# Run model comparison
# -----------------------------------------------------------------------
comp = mc.ModelComparison(
    models=[
        ("M1: 1-DOF",          problem_m1, prior_m1),
        ("M2: 2-DOF (correct)", problem_m2, prior_m2),
        ("M3: 2-DOF + mass",   problem_m3, prior_m3),
    ],
    tmcmc_kwargs={"n_particles": 1000, "n_mcmc_steps": 3},
)
comp.run()

print("=== Model comparison ===")
table = comp.summary()
print()
print(f"Best model: {comp.best_model()}")
print()

# -----------------------------------------------------------------------
# Individual Bayes factors (pairwise)
# -----------------------------------------------------------------------
log_evs = {r["name"]: r["log_evidence"] for r in table}
bf = mc.bayes_factor(log_evs["M2: 2-DOF (correct)"], log_evs["M1: 1-DOF"])
print(f"B(M2 vs M1): log10 BF = {bf['log10_bf']:.2f}  [{bf['evidence']}]")
bf32 = mc.bayes_factor(log_evs["M2: 2-DOF (correct)"], log_evs["M3: 2-DOF + mass"])
print(f"B(M2 vs M3): log10 BF = {bf32['log10_bf']:.2f}  [{bf32['evidence']}]")

# -----------------------------------------------------------------------
# Posterior of the correct model (M2)
# -----------------------------------------------------------------------
res_m2 = comp.get_result("M2: 2-DOF (correct)")
print()
print("=== M2 posterior ===")
print(f"  k1: {res_m2.mean()[0]:.3f} ± {res_m2.std()[0]:.3f}  (true: {K1_TRUE})")
print(f"  k2: {res_m2.mean()[1]:.3f} ± {res_m2.std()[1]:.3f}  (true: {K2_TRUE})")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    # Evidence comparison bar chart
    fig = comp.plot(title="Model class selection — 2-DOF shear frame")

    # Corner plot for M2
    fig2 = res_m2.plot_corner(
        true_values=[K1_TRUE, K2_TRUE],
        title="M2 posterior (k1, k2)",
    )

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
