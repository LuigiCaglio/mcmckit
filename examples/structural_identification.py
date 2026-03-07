"""Structural model updating: 2-DOF mass-spring system.

Identify two unknown stiffness values from noisy natural frequency measurements.

System layout:

    ___      k1       k2
   |   |---/\/\/---m2---/\/\/---|
   |   |                       |
   |___|---/\/\/---m1           |
         k0 (grnd)             wall

Simplified to a standard 2-DOF chain:

   ground---k1---[m1]---k2---[m2]---k3---wall

   m1 = m2 = 1.0  (known)
   k1, k2, k3    (unknown, 3 parameters)

Forward model: eigenvalue problem  K v = ω² M v  -> natural frequencies ω

Three noise modes are compared:
  A — fixed sigma (sigma known from sensor specs)
  B — estimated sigma (log_sigma sampled jointly with stiffness)
  C — marginalised sigma (InvGamma prior, sigma integrated out analytically)
"""

import numpy as np
import mcmckit as mc

rng = np.random.default_rng(7)

# -----------------------------------------------------------------------
# System definition
# -----------------------------------------------------------------------
M1, M2 = 1.0, 1.0          # known masses
TRUE_K = np.array([8.0, 5.0, 6.0])   # k1, k2, k3  (to be identified)
TRUE_SIGMA = 0.05           # measurement noise on natural frequencies (rad/s)

N_REPEAT = 4                # repeated measurements of each frequency


def stiffness_matrix(k):
    """Assembled stiffness matrix for the 2-DOF chain."""
    k1, k2, k3 = k
    return np.array([
        [k1 + k2,    -k2   ],
        [  -k2,    k2 + k3 ],
    ])


def natural_frequencies(k):
    """Return sorted natural frequencies [ω1, ω2] in rad/s."""
    K = stiffness_matrix(k)
    M = np.diag([M1, M2])
    # solve generalised eigenvalue: K v = ω² M v
    # -> M^{-0.5} K M^{-0.5} v' = ω² v'
    Minv_sqrt = np.diag(1.0 / np.sqrt([M1, M2]))
    A = Minv_sqrt @ K @ Minv_sqrt
    eigvals = np.linalg.eigvalsh(A)        # symmetric -> real, sorted ascending
    return np.sqrt(np.maximum(eigvals, 0))  # ω = sqrt(λ)


# -----------------------------------------------------------------------
# Synthetic observations: N_REPEAT measurements of [ω1, ω2]
# -----------------------------------------------------------------------
omega_true = natural_frequencies(TRUE_K)
print(f"True natural frequencies:  ω1 = {omega_true[0]:.4f},  ω2 = {omega_true[1]:.4f}  rad/s")
print(f"True stiffness:  k1={TRUE_K[0]}, k2={TRUE_K[1]}, k3={TRUE_K[2]}")
print()

# Stack repeated measurements into a flat vector: [ω1_m1, ω1_m2, ..., ω2_m1, ...]
y_obs = np.tile(omega_true, N_REPEAT) + rng.normal(scale=TRUE_SIGMA, size=2 * N_REPEAT)


def forward_model(theta_model):
    """Predict repeated frequency measurements for given stiffness."""
    return np.tile(natural_frequencies(theta_model), N_REPEAT)


# -----------------------------------------------------------------------
# Prior: k1, k2, k3 > 0,  log-uniform on [1, 30]
# -----------------------------------------------------------------------
K_LO, K_HI = 1.0, 30.0

def log_prior_model(theta):
    if np.any(theta <= K_LO) or np.any(theta >= K_HI):
        return -np.inf
    return 0.0   # flat on (K_LO, K_HI)^3


INIT_COV = np.diag([0.5, 0.5, 0.5])
X0 = [6.0, 4.0, 5.0]     # initial guess (deliberately off from truth)

# -----------------------------------------------------------------------
# Mode A — fixed sigma
# -----------------------------------------------------------------------
ll_A = mc.GaussianNoiseLikelihood(forward_model, y_obs, noise_std=TRUE_SIGMA)
prob_A = mc.Problem(prior=log_prior_model, likelihood=ll_A,
                    param_names=["k1", "k2", "k3"])

res_A = mc.DRAM(n_samples=15_000, initial_cov=INIT_COV).run(prob_A, x0=X0).discard(3000)

# -----------------------------------------------------------------------
# Mode B — estimated sigma
# -----------------------------------------------------------------------
def log_prior_B(theta):
    if np.any(theta[:3] <= K_LO) or np.any(theta[:3] >= K_HI):
        return -np.inf
    sigma = np.exp(float(theta[3]))
    return -0.5 * sigma**2   # HalfNormal(1) on sigma

ll_B = mc.GaussianNoiseLikelihood(forward_model, y_obs)   # noise_std=None
prob_B = mc.Problem(prior=log_prior_B, likelihood=ll_B,
                    param_names=["k1", "k2", "k3", "log_sigma"])

res_B = mc.DRAM(n_samples=15_000,
                initial_cov=np.diag([0.5, 0.5, 0.5, 0.1])).run(
    prob_B, x0=X0 + [np.log(0.1)]).discard(3000)

sigma_post_B = np.exp(res_B.samples[:, 3])

# -----------------------------------------------------------------------
# Mode C — marginalised sigma (InvGamma prior on sigma^2)
# -----------------------------------------------------------------------
ll_C = mc.GaussianNoiseLikelihood(forward_model, y_obs,
                                  marginalise_noise=True,
                                  inv_gamma_alpha=2.0,
                                  inv_gamma_beta=TRUE_SIGMA**2)
prob_C = mc.Problem(prior=log_prior_model, likelihood=ll_C,
                    param_names=["k1", "k2", "k3"])

res_C = mc.DRAM(n_samples=15_000, initial_cov=INIT_COV).run(prob_C, x0=X0).discard(3000)

sigma_C_samples = np.array([ll_C.posterior_sigma(s) for s in res_C.samples[::20]])

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------
w = 12
hdr = f"{'Mode':>16}  {'k1':>{w}}  {'k2':>{w}}  {'k3':>{w}}  {'sigma':>{w}}"
sep = "-" * len(hdr)
print(hdr)
print(sep)
print(f"{'True':>16}  {TRUE_K[0]:>{w}.4f}  {TRUE_K[1]:>{w}.4f}  {TRUE_K[2]:>{w}.4f}  {TRUE_SIGMA:>{w}.4f}")
print(f"{'A — fixed':>16}  {res_A.mean()[0]:>{w}.4f}  {res_A.mean()[1]:>{w}.4f}  {res_A.mean()[2]:>{w}.4f}  {'(fixed)':>{w}}")
print(f"{'B — estimated':>16}  {res_B.mean()[0]:>{w}.4f}  {res_B.mean()[1]:>{w}.4f}  {res_B.mean()[2]:>{w}.4f}  {sigma_post_B.mean():>{w}.4f}")
print(f"{'C — marginalised':>16}  {res_C.mean()[0]:>{w}.4f}  {res_C.mean()[1]:>{w}.4f}  {res_C.mean()[2]:>{w}.4f}  {sigma_C_samples.mean():>{w}.4f}")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    res_A.plot_corner(true_values=TRUE_K, title="Mode A: fixed sigma")
    res_B.plot_corner(
        true_values=list(TRUE_K) + [np.log(TRUE_SIGMA)],
        title="Mode B: estimated sigma  (last param = log σ)")
    res_C.plot_corner(true_values=TRUE_K, title="Mode C: marginalised sigma")

    # Posterior predictive on natural frequencies
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    omega_labels = ["ω₁", "ω₂"]
    colors = {"A": "steelblue", "B": "darkorange", "C": "seagreen"}

    for ax, freq_idx in zip(axes, [0, 1]):
        for mode, res, label in [("A", res_A, "fixed σ"),
                                  ("B", res_B, "estimated σ"),
                                  ("C", res_C, "marginalised σ")]:
            omegas = np.array([natural_frequencies(s[:3])[freq_idx]
                                for s in res.samples[::10]])
            ax.hist(omegas, bins=40, density=True, alpha=0.5,
                    color=colors[mode], label=f"Mode {mode}: {label}")
        ax.axvline(omega_true[freq_idx], color="crimson", lw=1.5,
                   ls="--", label=f"true {omega_labels[freq_idx]}")
        obs_i = y_obs[freq_idx::2] if freq_idx < 2 else y_obs[freq_idx::2]
        # actual observed values for this frequency slot
        obs_vals = y_obs[freq_idx * N_REPEAT: (freq_idx + 1) * N_REPEAT]
        ax.scatter(obs_vals, np.zeros(N_REPEAT) - 0.5, marker="|",
                   s=200, color="k", zorder=5, label="observations" if freq_idx == 0 else None)
        ax.set_xlabel(f"{omega_labels[freq_idx]}  (rad/s)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.set_title(f"Posterior predictive — {omega_labels[freq_idx]}")
        if freq_idx == 0:
            ax.legend(fontsize=7)

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
