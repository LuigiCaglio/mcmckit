"""Sequential Bayesian updating — using the previous posterior as the next prior.

Scenario
--------
A 2-DOF shear frame is monitored over three measurement campaigns:

  Campaign 0 (baseline):  flat prior, 5 noisy frequency measurements
  Campaign 1 (new data):  posterior from 0 becomes prior, 5 more measurements
  Campaign 2 (new data):  posterior from 1 becomes prior, 10 measurements

The posterior uncertainty should shrink with each new dataset, converging to
the true stiffness values.

Two methods are compared:
  - Gaussian approximation  (fast, any dimension)
  - KDE                     (non-parametric, more general)

The final posterior from sequential updating is also compared to a single
batch run using all data at once — both should agree (Bayes consistency).

True system:  k1 = 10 N/m,  k2 = 8 N/m,  m1 = m2 = 1 kg
"""

import numpy as np
import mcmckit as mc
from scipy.linalg import eigh

rng = np.random.default_rng(7)

TRUE_K1 = 10.0
TRUE_K2 = 8.0
SIGMA_OBS = 0.05   # rad/s


# ---------------------------------------------------------------------------
# Forward model: 2-DOF shear frame natural frequencies
# ---------------------------------------------------------------------------
def natural_frequencies(k1, k2, m1=1.0, m2=1.0):
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    M = np.diag([m1, m2])
    vals, _ = eigh(K, M)
    return np.sqrt(np.maximum(vals, 0.0))


def log_prior_flat(theta):
    k1, k2 = theta
    if k1 <= 0 or k2 <= 0:
        return -np.inf
    return (-0.5 * ((np.log(k1) - np.log(10)) / 2.0) ** 2
            - 0.5 * ((np.log(k2) - np.log(10)) / 2.0) ** 2)


true_freqs = natural_frequencies(TRUE_K1, TRUE_K2)
print(f"True: k1={TRUE_K1}, k2={TRUE_K2}")
print(f"True frequencies: f1={true_freqs[0]:.4f}, f2={true_freqs[1]:.4f} rad/s")
print()


# ---------------------------------------------------------------------------
# Helper: build a likelihood for N_obs measurements per mode
# ---------------------------------------------------------------------------
def make_likelihood(n_obs, seed=None):
    _rng = np.random.default_rng(seed)
    y = np.concatenate([
        _rng.normal(true_freqs[0], SIGMA_OBS, size=n_obs),
        _rng.normal(true_freqs[1], SIGMA_OBS, size=n_obs),
    ])
    return mc.GaussianNoiseLikelihood(
        forward_model=lambda theta: np.repeat(
            natural_frequencies(theta[0], theta[1]), n_obs
        ),
        y_obs=y,
        noise_std=SIGMA_OBS,
    )


# ---------------------------------------------------------------------------
# Sampler settings
# ---------------------------------------------------------------------------
SAMPLER = mc.DRAM(n_samples=12_000, initial_cov=0.5 * np.eye(2))
DISCARD = 2000
PARAM_NAMES = ["k1", "k2"]


# ---------------------------------------------------------------------------
# Campaign 0: flat prior
# ---------------------------------------------------------------------------
print("=== Campaign 0: flat prior, 5 observations ===")
ll0 = make_likelihood(n_obs=5, seed=10)
prob0 = mc.Problem(prior=log_prior_flat, likelihood=ll0, param_names=PARAM_NAMES)
result0 = SAMPLER.run(prob0, x0=[10.0, 8.0])
r0 = result0.discard(DISCARD)
print(f"  k1 = {r0.mean()[0]:.3f} +/- {r0.std()[0]:.3f}  (true: {TRUE_K1})")
print(f"  k2 = {r0.mean()[1]:.3f} +/- {r0.std()[1]:.3f}  (true: {TRUE_K2})")
print()


# ---------------------------------------------------------------------------
# Campaign 1: posterior from 0 as prior — Gaussian method
# ---------------------------------------------------------------------------
print("=== Campaign 1 (Gaussian prior): 5 new observations ===")
prior1_gauss = result0.as_prior(method='gaussian', discard=DISCARD)
print(f"  PosteriorPrior: {prior1_gauss}")

ll1 = make_likelihood(n_obs=5, seed=20)
prob1_g = mc.Problem(prior=prior1_gauss, likelihood=ll1, param_names=PARAM_NAMES)
result1_g = mc.DRAM(
    n_samples=12_000, initial_cov=prior1_gauss.cov
).run(prob1_g, x0=prior1_gauss.mean)
r1_g = result1_g.discard(DISCARD)
print(f"  k1 = {r1_g.mean()[0]:.3f} +/- {r1_g.std()[0]:.3f}  (true: {TRUE_K1})")
print(f"  k2 = {r1_g.mean()[1]:.3f} +/- {r1_g.std()[1]:.3f}  (true: {TRUE_K2})")
print()


# ---------------------------------------------------------------------------
# Campaign 1: same data, KDE prior
# ---------------------------------------------------------------------------
print("=== Campaign 1 (KDE prior): same 5 new observations ===")
prior1_kde = result0.as_prior(method='kde', discard=DISCARD)
print(f"  PosteriorPrior: {prior1_kde}")

prob1_k = mc.Problem(prior=prior1_kde, likelihood=ll1, param_names=PARAM_NAMES)
result1_k = mc.DRAM(
    n_samples=12_000, initial_cov=prior1_kde.cov
).run(prob1_k, x0=prior1_kde.mean)
r1_k = result1_k.discard(DISCARD)
print(f"  k1 = {r1_k.mean()[0]:.3f} +/- {r1_k.std()[0]:.3f}  (true: {TRUE_K1})")
print(f"  k2 = {r1_k.mean()[1]:.3f} +/- {r1_k.std()[1]:.3f}  (true: {TRUE_K2})")
print()


# ---------------------------------------------------------------------------
# Campaign 2: further update (Gaussian chain), 10 new observations
# ---------------------------------------------------------------------------
print("=== Campaign 2 (Gaussian): 10 new observations ===")
prior2 = result1_g.as_prior(method='gaussian', discard=DISCARD)
ll2 = make_likelihood(n_obs=10, seed=30)
prob2 = mc.Problem(prior=prior2, likelihood=ll2, param_names=PARAM_NAMES)
result2 = mc.DRAM(
    n_samples=12_000, initial_cov=prior2.cov
).run(prob2, x0=prior2.mean)
r2 = result2.discard(DISCARD)
print(f"  k1 = {r2.mean()[0]:.3f} +/- {r2.std()[0]:.3f}  (true: {TRUE_K1})")
print(f"  k2 = {r2.mean()[1]:.3f} +/- {r2.std()[1]:.3f}  (true: {TRUE_K2})")
print()


# ---------------------------------------------------------------------------
# Batch reference: all data combined, flat prior
# ---------------------------------------------------------------------------
print("=== Batch reference: all 20 observations, flat prior ===")
# Recreate the same datasets
y_all = np.concatenate([
    np.concatenate([
        np.random.default_rng(s).normal(true_freqs[0], SIGMA_OBS, size=n)
        for s, n in [(10, 5), (20, 5), (30, 10)]
    ]),
    np.concatenate([
        np.random.default_rng(s).normal(true_freqs[1], SIGMA_OBS, size=n)
        for s, n in [(10, 5), (20, 5), (30, 10)]
    ]),
])
ll_batch = mc.GaussianNoiseLikelihood(
    forward_model=lambda theta: np.repeat(
        natural_frequencies(theta[0], theta[1]), 20
    ),
    y_obs=y_all,
    noise_std=SIGMA_OBS,
)
prob_batch = mc.Problem(prior=log_prior_flat, likelihood=ll_batch, param_names=PARAM_NAMES)
result_batch = SAMPLER.run(prob_batch, x0=[10.0, 8.0])
r_batch = result_batch.discard(DISCARD)
print(f"  k1 = {r_batch.mean()[0]:.3f} +/- {r_batch.std()[0]:.3f}  (true: {TRUE_K1})")
print(f"  k2 = {r_batch.mean()[1]:.3f} +/- {r_batch.std()[1]:.3f}  (true: {TRUE_K2})")
print()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("Summary — posterior uncertainty (std) across stages:")
print(f"  {'Stage':<30} {'k1 std':>8}  {'k2 std':>8}")
print(f"  {'-'*50}")
for label, r in [
    ("Campaign 0 (flat prior, 5 obs)", r0),
    ("Campaign 1 (Gaussian prior, +5)", r1_g),
    ("Campaign 1 (KDE prior, +5)",     r1_k),
    ("Campaign 2 (Gaussian prior, +10)", r2),
    ("Batch (flat prior, 20 obs)",     r_batch),
]:
    print(f"  {label:<40} {r.std()[0]:>8.4f}  {r.std()[1]:>8.4f}")
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    stages = [
        ("Campaign 0\n(flat prior, 5 obs)", r0),
        ("Campaign 1\n(Gaussian prior, +5 obs)", r1_g),
        ("Campaign 2\n(Gaussian prior, +10 obs)", r2),
    ]
    for ax, (label, r) in zip(axes, stages):
        ax.scatter(r.samples[:, 0], r.samples[:, 1], s=1, alpha=0.2, rasterized=True)
        ax.axvline(TRUE_K1, color="r", lw=1.5, label=f"True k1={TRUE_K1}")
        ax.axhline(TRUE_K2, color="g", lw=1.5, label=f"True k2={TRUE_K2}")
        ax.set_xlabel("k1 [N/m]")
        ax.set_title(label)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("k2 [N/m]")
    plt.suptitle("Sequential Bayesian updating — posterior shrinks with each campaign")
    plt.tight_layout()
    plt.savefig("sequential_posterior.png", dpi=150)
    print("Saved: sequential_posterior.png")
    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
