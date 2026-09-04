"""Structural model updating: 2-DOF mass-spring system.

Identify three unknown stiffness values from noisy natural frequency
measurements.  The two frequencies are measured with *different* noise levels
(ω1 is harder to measure than ω2), which is the typical situation in SHM.

System:   ground---k1---[m1]---k2---[m2]---k3---wall
          m1 = m2 = 1.0  (known)
          k1, k2, k3    (unknown, to be identified)

Forward model: eigenvalue problem  K v = ω² M v  ->  [ω1, ω2]

N_REPEAT measurements are taken for each frequency, giving a flat observation
vector:  y = [ω1_obs×N_REPEAT, ω2_obs×N_REPEAT]

Groups:
  channel 0 = indices 0..N_REPEAT-1          (all ω1 measurements)
  channel 1 = indices N_REPEAT..2*N_REPEAT-1 (all ω2 measurements)

Three noise modes are compared:
  A — fixed per-channel sigma  [0.08, 0.03]
  B — estimated per-channel sigma  (2 free log_σ params at end of theta)
  C — marginalised per-channel sigma  (independent InvGamma priors)
"""

import numpy as np
import mcmckit as mc

rng = np.random.default_rng(7)

# -----------------------------------------------------------------------
# System + true values
# -----------------------------------------------------------------------
M1, M2 = 1.0, 1.0
TRUE_K = np.array([8.0, 5.0, 6.0])
TRUE_SIGMA = np.array([0.08, 0.03])   # different noise per frequency
N_REPEAT = 6


def natural_frequencies(k):
    k1, k2, k3 = k
    K = np.array([[k1 + k2, -k2], [-k2, k2 + k3]])
    Minvhalf = np.diag(1.0 / np.sqrt([M1, M2]))
    eigvals = np.linalg.eigvalsh(Minvhalf @ K @ Minvhalf)
    return np.sqrt(np.maximum(eigvals, 0.0))


omega_true = natural_frequencies(TRUE_K)
print(f"True natural frequencies:  omega1={omega_true[0]:.4f},  omega2={omega_true[1]:.4f}  rad/s")
print(f"True stiffness:  k={TRUE_K}")
print(f"True sigma:      sigma={TRUE_SIGMA}  (per channel)\n")

# Flat observation vector: [ω1×N_REPEAT, ω2×N_REPEAT]
y_obs = np.concatenate([
    np.repeat(omega_true[0], N_REPEAT) + rng.normal(scale=TRUE_SIGMA[0], size=N_REPEAT),
    np.repeat(omega_true[1], N_REPEAT) + rng.normal(scale=TRUE_SIGMA[1], size=N_REPEAT),
])

# Channel definitions
groups = [
    np.arange(0, N_REPEAT),              # channel 0: all ω1 measurements
    np.arange(N_REPEAT, 2 * N_REPEAT),   # channel 1: all ω2 measurements
]


def forward_model(theta_model):
    return np.concatenate([
        np.repeat(natural_frequencies(theta_model)[0], N_REPEAT),
        np.repeat(natural_frequencies(theta_model)[1], N_REPEAT),
    ])


# -----------------------------------------------------------------------
# Prior on model params
# -----------------------------------------------------------------------
K_LO, K_HI = 1.0, 30.0

def log_prior_model(theta):
    if np.any(theta <= K_LO) or np.any(theta >= K_HI):
        return -np.inf
    return 0.0


INIT_COV_3 = np.diag([0.5, 0.5, 0.5])
X0 = [6.0, 4.0, 5.0]

# -----------------------------------------------------------------------
# Mode A — fixed per-channel sigma
# -----------------------------------------------------------------------
ll_A = mc.GaussianNoiseLikelihood(
    forward_model, y_obs,
    noise_std=TRUE_SIGMA,      # array of length n_channels=2
    groups=groups,
)
prob_A = mc.Problem(prior=log_prior_model, likelihood=ll_A,
                    param_names=["k1", "k2", "k3"])

res_A = mc.DRAM(n_samples=15_000, initial_cov=INIT_COV_3).run(
    prob_A, x0=X0).discard(3000)

# -----------------------------------------------------------------------
# Mode B — estimated per-channel sigma
# theta = [k1, k2, k3, log_sigma_1, log_sigma_2]
# -----------------------------------------------------------------------
def log_prior_B(theta):
    if np.any(theta[:3] <= K_LO) or np.any(theta[:3] >= K_HI):
        return -np.inf
    sigmas = np.exp(theta[3:])
    return -0.5 * np.dot(sigmas, sigmas)   # HalfNormal(1) on each sigma

ll_B = mc.GaussianNoiseLikelihood(forward_model, y_obs, groups=groups)
prob_B = mc.Problem(prior=log_prior_B, likelihood=ll_B,
                    param_names=["k1", "k2", "k3", "log_σ₁", "log_σ₂"])

x0_B = X0 + [np.log(0.1), np.log(0.05)]
res_B = mc.DRAM(n_samples=15_000,
                initial_cov=np.diag([0.5, 0.5, 0.5, 0.1, 0.1])).run(
    prob_B, x0=x0_B).discard(3000)

sigma_post_B = np.exp(res_B.samples[:, 3:])   # shape (n_samples, 2)

# -----------------------------------------------------------------------
# Mode C — marginalised per-channel sigma
# -----------------------------------------------------------------------
ll_C = mc.GaussianNoiseLikelihood(
    forward_model, y_obs,
    groups=groups,
    marginalise_noise=True,
    inv_gamma_alpha=[2.0, 2.0],
    inv_gamma_beta=[TRUE_SIGMA[0]**2, TRUE_SIGMA[1]**2],
)
prob_C = mc.Problem(prior=log_prior_model, likelihood=ll_C,
                    param_names=["k1", "k2", "k3"])

res_C = mc.DRAM(n_samples=15_000, initial_cov=INIT_COV_3).run(
    prob_C, x0=X0).discard(3000)

sigma_post_C = np.array([ll_C.posterior_sigma(s) for s in res_C.samples[::20]])

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
w = 10
print(f"{'Mode':>18}  {'k1':>{w}}  {'k2':>{w}}  {'k3':>{w}}  {'sigma1':>{w}}  {'sigma2':>{w}}")
print("-" * (18 + 5 * (w + 2)))
print(f"{'True':>18}  {TRUE_K[0]:>{w}.4f}  {TRUE_K[1]:>{w}.4f}  {TRUE_K[2]:>{w}.4f}  {TRUE_SIGMA[0]:>{w}.4f}  {TRUE_SIGMA[1]:>{w}.4f}")
print(f"{'A - fixed':>18}  {res_A.mean()[0]:>{w}.4f}  {res_A.mean()[1]:>{w}.4f}  {res_A.mean()[2]:>{w}.4f}  {'(fixed)':>{w}}  {'(fixed)':>{w}}")
print(f"{'B - estimated':>18}  {res_B.mean()[0]:>{w}.4f}  {res_B.mean()[1]:>{w}.4f}  {res_B.mean()[2]:>{w}.4f}  {sigma_post_B[:,0].mean():>{w}.4f}  {sigma_post_B[:,1].mean():>{w}.4f}")
print(f"{'C - marginalised':>18}  {res_C.mean()[0]:>{w}.4f}  {res_C.mean()[1]:>{w}.4f}  {res_C.mean()[2]:>{w}.4f}  {sigma_post_C[:,0].mean():>{w}.4f}  {sigma_post_C[:,1].mean():>{w}.4f}")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    res_A.plot_corner(true_values=TRUE_K, title="Mode A: fixed per-channel σ")
    res_B.plot_corner(
        true_values=list(TRUE_K) + list(np.log(TRUE_SIGMA)),
        title="Mode B: estimated per-channel σ")
    res_C.plot_corner(true_values=TRUE_K, title="Mode C: marginalised per-channel σ")

    # Posterior predictive check
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    freq_labels = ["ω₁", "ω₂"]
    clr = {"A": "steelblue", "B": "darkorange", "C": "seagreen"}
    for ax, fi in zip(axes, [0, 1]):
        for mode, res in [("A", res_A), ("B", res_B), ("C", res_C)]:
            omegas = np.array([natural_frequencies(s[:3])[fi]
                               for s in res.samples[::10]])
            ax.hist(omegas, bins=40, density=True, alpha=0.5,
                    color=clr[mode], label=f"Mode {mode}")
        ax.axvline(omega_true[fi], color="crimson", lw=1.5, ls="--",
                   label="true")
        obs_vals = y_obs[fi * N_REPEAT: (fi + 1) * N_REPEAT]
        ax.scatter(obs_vals, np.zeros(N_REPEAT) - 0.5,
                   marker="|", s=200, color="k", zorder=5,
                   label="obs" if fi == 0 else None)
        ax.set_xlabel(f"{freq_labels[fi]}  (rad/s)")
        ax.set_ylabel("density")
        ax.set_title(f"Posterior predictive — {freq_labels[fi]}")
        if fi == 0:
            ax.legend(fontsize=8)

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
