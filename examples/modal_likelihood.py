"""Modal likelihood: Bayesian updating with frequencies AND mode shapes.

Demonstrates how to use ModalLikelihood for structural model updating when
both natural frequency and mode shape data are available.

Scenario
--------
2-DOF shear frame (lumped masses m1 = m2 = 1 kg).  We observe:
  - 2 natural frequencies  (circular, rad/s)
  - 2 mode shapes          (2 sensor DOFs — full instrumentation)

We identify stiffnesses k1 and k2.

True system:  k1 = 10 N/m,  k2 = 8 N/m
Observations: 10 realisations of each frequency + one noisy mode shape per mode

The example compares three likelihood formulations:
  1. Frequencies only        (GaussianNoiseLikelihood)
  2. Mode shapes only        (ModalLikelihood, sigma_freq → ∞)
  3. Frequencies + shapes    (ModalLikelihood, combined)
"""

import numpy as np
import mcmckit as mc
from scipy.linalg import eigh

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# True system
# ---------------------------------------------------------------------------
TRUE_K1 = 10.0
TRUE_K2 = 8.0
M1 = M2 = 1.0
SIGMA_FREQ = 0.05   # rad/s
SIGMA_SHAPE = 0.02  # additive noise on each mode shape component
N_FREQ_OBS = 10     # repeated frequency measurements per mode


def system_modes(k1, k2, m1=1.0, m2=1.0):
    """Return (circular frequencies [rad/s], mode shape matrix [n_dof, n_modes])."""
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    M = np.diag([m1, m2])
    vals, vecs = eigh(K, M)
    freqs = np.sqrt(np.maximum(vals, 0.0))
    # Normalise so that the largest absolute component of each shape = 1
    for j in range(vecs.shape[1]):
        vecs[:, j] /= vecs[np.argmax(np.abs(vecs[:, j])), j]
    return freqs, vecs


true_freqs, true_shapes = system_modes(TRUE_K1, TRUE_K2)

print("True system")
print(f"  k1 = {TRUE_K1},  k2 = {TRUE_K2}")
print(f"  f1 = {true_freqs[0]:.4f} rad/s,  f2 = {true_freqs[1]:.4f} rad/s")
print(f"  mode1 = {true_shapes[:, 0]}")
print(f"  mode2 = {true_shapes[:, 1]}")
print()

# ---------------------------------------------------------------------------
# Simulated observations
# ---------------------------------------------------------------------------
freq_obs = np.array([
    rng.normal(true_freqs[0], SIGMA_FREQ, size=N_FREQ_OBS).mean(),
    rng.normal(true_freqs[1], SIGMA_FREQ, size=N_FREQ_OBS).mean(),
])

# Single mode shape observation per mode (common in practice)
shapes_obs = true_shapes + rng.normal(0, SIGMA_SHAPE, size=true_shapes.shape)
# Re-normalise observed shapes (mass-normalisation free, just scale to max=1)
for j in range(shapes_obs.shape[1]):
    shapes_obs[:, j] /= shapes_obs[np.argmax(np.abs(shapes_obs[:, j])), j]

print("Observed data")
print(f"  f1_obs = {freq_obs[0]:.4f},  f2_obs = {freq_obs[1]:.4f}  (true: {true_freqs[0]:.4f}, {true_freqs[1]:.4f})")
print(f"  mode1_obs = {shapes_obs[:, 0]}  (true: {true_shapes[:, 0]})")
print(f"  mode2_obs = {shapes_obs[:, 1]}  (true: {true_shapes[:, 1]})")
print()

# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
def forward(theta):
    k1, k2 = theta
    if k1 <= 0 or k2 <= 0:
        return np.array([np.nan, np.nan]), np.zeros((2, 2))
    return system_modes(k1, k2)


def log_prior(theta):
    k1, k2 = theta
    if k1 <= 0 or k2 <= 0:
        return -np.inf
    # Lognormal prior centred on 10 N/m with large spread
    return (
        -0.5 * ((np.log(k1) - np.log(10)) / 1.5) ** 2
        - 0.5 * ((np.log(k2) - np.log(10)) / 1.5) ** 2
    )


# ---------------------------------------------------------------------------
# Formulation 1: frequencies only
# ---------------------------------------------------------------------------
print("=== Formulation 1: Frequencies only ===")

# Flatten N_FREQ_OBS realisations per mode as grouped observations
y_freq_all = np.concatenate([
    rng.normal(true_freqs[0], SIGMA_FREQ, size=N_FREQ_OBS),
    rng.normal(true_freqs[1], SIGMA_FREQ, size=N_FREQ_OBS),
])
ll_freq = mc.GaussianNoiseLikelihood(
    forward_model=lambda theta: np.repeat(forward(theta)[0], N_FREQ_OBS),
    y_obs=y_freq_all,
    noise_std=SIGMA_FREQ,
)
prob_freq = mc.Problem(prior=log_prior, likelihood=ll_freq, param_names=["k1", "k2"])

result_freq = mc.DRAM(n_samples=15_000, initial_cov=0.5 * np.eye(2)).run(
    prob_freq, x0=[10.0, 8.0]
)
r_freq = result_freq.discard(3000)
print(f"  k1 = {r_freq.mean()[0]:.3f} ± {r_freq.std()[0]:.3f}  (true: {TRUE_K1})")
print(f"  k2 = {r_freq.mean()[1]:.3f} ± {r_freq.std()[1]:.3f}  (true: {TRUE_K2})")
print()

# ---------------------------------------------------------------------------
# Formulation 2: combined (frequencies + mode shapes) — ModalLikelihood
# ---------------------------------------------------------------------------
print("=== Formulation 2: Frequencies + mode shapes (ModalLikelihood) ===")

ll_modal = mc.ModalLikelihood(
    forward_model=forward,
    freq_obs=freq_obs,
    mode_shapes_obs=shapes_obs,
    sigma_freq=SIGMA_FREQ,
    sigma_mac=0.05,          # tolerance on MAC deviation from 1
    auto_pair=True,
    freq_error="relative",
)
prob_modal = mc.Problem(prior=log_prior, likelihood=ll_modal, param_names=["k1", "k2"])

result_modal = mc.DRAM(n_samples=15_000, initial_cov=0.5 * np.eye(2)).run(
    prob_modal, x0=[10.0, 8.0]
)
r_modal = result_modal.discard(3000)
print(f"  k1 = {r_modal.mean()[0]:.3f} ± {r_modal.std()[0]:.3f}  (true: {TRUE_K1})")
print(f"  k2 = {r_modal.mean()[1]:.3f} ± {r_modal.std()[1]:.3f}  (true: {TRUE_K2})")
print()

# MAC values at posterior mean
theta_map = r_modal.mean()
mac_vals = ll_modal.mac_values(theta_map)
freq_errs = ll_modal.freq_errors(theta_map)
print(f"  At posterior mean theta = [{theta_map[0]:.3f}, {theta_map[1]:.3f}]:")
print(f"    MAC values:      {mac_vals[0]:.4f},  {mac_vals[1]:.4f}")
print(f"    Freq errors (%): {freq_errs[0]*100:.3f},  {freq_errs[1]*100:.3f}")
print()

# MAC matrix at posterior mean
print("  MAC matrix (observed vs predicted at posterior mean):")
_, shapes_at_mean = forward(theta_map)
M = mc.mac_matrix(shapes_obs, shapes_at_mean)
print(f"    [[{M[0,0]:.4f}, {M[0,1]:.4f}],")
print(f"     [{M[1,0]:.4f}, {M[1,1]:.4f}]]")
print()

# ---------------------------------------------------------------------------
# Formulation 3: free frequency noise
# ---------------------------------------------------------------------------
print("=== Formulation 3: ModalLikelihood with free frequency noise ===")

ll_free = mc.ModalLikelihood(
    forward_model=forward,
    freq_obs=freq_obs,
    mode_shapes_obs=shapes_obs,
    sigma_freq=None,        # free — log_sigma appended to theta
    sigma_mac=0.05,
    auto_pair=True,
    freq_error="relative",
)
# theta = [k1, k2, log_sigma_freq_0, log_sigma_freq_1]
x0_free = [10.0, 8.0, np.log(0.05), np.log(0.05)]
def log_prior_free(theta):
    k1, k2 = theta[0], theta[1]
    log_sf1, log_sf2 = theta[2], theta[3]
    if k1 <= 0 or k2 <= 0:
        return -np.inf
    # Stiffness: lognormal centred on 10 N/m
    lp = (
        -0.5 * ((np.log(k1) - np.log(10)) / 1.5) ** 2
        - 0.5 * ((np.log(k2) - np.log(10)) / 1.5) ** 2
    )
    # Informative prior on log_sigma_freq: N(log(0.05), 0.5^2)
    # prevents sigma -> inf degeneracy
    lp -= 0.5 * ((log_sf1 - np.log(SIGMA_FREQ)) / 0.5) ** 2
    lp -= 0.5 * ((log_sf2 - np.log(SIGMA_FREQ)) / 0.5) ** 2
    return lp

prob_free = mc.Problem(
    prior=log_prior_free,
    likelihood=ll_free,
    param_names=["k1", "k2", "log_sf1", "log_sf2"],
)
result_free = mc.DRAM(n_samples=15_000, initial_cov=0.3 * np.eye(4)).run(
    prob_free, x0=x0_free
)
r_free = result_free.discard(3000)
print(f"  k1         = {r_free.mean()[0]:.3f} ± {r_free.std()[0]:.3f}  (true: {TRUE_K1})")
print(f"  k2         = {r_free.mean()[1]:.3f} ± {r_free.std()[1]:.3f}  (true: {TRUE_K2})")
print(f"  sigma_f1   = {np.exp(r_free.mean()[2]):.4f} +/- {r_free.std()[2]:.4f} log-scale  (true: {SIGMA_FREQ})")
print(f"  sigma_f2   = {np.exp(r_free.mean()[3]):.4f} +/- {r_free.std()[3]:.4f} log-scale  (true: {SIGMA_FREQ})")
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, r, label in zip(
        axes,
        [r_freq, r_modal],
        ["Frequencies only", "Freq + mode shapes (MAC)"],
    ):
        samples = r.samples
        ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, rasterized=True)
        ax.axvline(TRUE_K1, color="r", lw=1.5, label=f"True k1={TRUE_K1}")
        ax.axhline(TRUE_K2, color="g", lw=1.5, label=f"True k2={TRUE_K2}")
        ax.set_xlabel("k1 [N/m]")
        ax.set_ylabel("k2 [N/m]")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("modal_posterior_comparison.png", dpi=150)
    print("Saved: modal_posterior_comparison.png")

    # Corner plot for combined formulation
    fig2 = r_modal.plot_corner(true_values=[TRUE_K1, TRUE_K2],
                               title="Posterior: Freq + mode shapes")
    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
