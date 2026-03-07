"""Sequential Bayesian updating — 10-parameter structural identification.

Identifies the 10 inter-story stiffnesses of a 10-story shear building from
repeated natural frequency measurements arriving in three monitoring campaigns.

This example demonstrates:
  - Sequential updating scales cleanly to higher dimensions
  - The Gaussian PosteriorPrior is the practical choice for >= 6 parameters
    (KDE degrades in quality and speed with dimension)
  - Posterior uncertainty shrinks with each new dataset
  - Sequential and batch results agree (Bayes consistency)

System
------
10-story shear frame, masses m_i = 1 kg, inter-story stiffnesses k_1..k_10.
True values: most floors at 10 N/m, floors 3 and 7 slightly softer at 8 N/m.

Observations
------------
7 natural frequencies (out of 10 modes) per campaign, each measured 6 times
with sigma_obs = 0.03 rad/s.  Three campaigns = 18 frequency values per mode.
"""

import numpy as np
import mcmckit as mc
from scipy.linalg import eigh

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# System definition
# ---------------------------------------------------------------------------
N_FLOORS = 10
N_MODES_OBS = 10     # observe all modes (fully determined with tight prior)
N_REPS = 6           # repeated frequency measurements per campaign per mode
SIGMA_OBS = 0.03     # rad/s

TRUE_K = np.array([10., 10., 8., 10., 10., 10., 8., 10., 10., 10.])
MASS = np.ones(N_FLOORS)


def build_stiffness_matrix(k):
    """Tridiagonal stiffness matrix for a shear building."""
    n = len(k)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = k[i]
        if i < n - 1:
            K[i, i] += k[i + 1]
            K[i, i + 1] = -k[i + 1]
            K[i + 1, i] = -k[i + 1]
    return K


def natural_frequencies(k):
    """Return all natural frequencies (rad/s) sorted ascending."""
    K = build_stiffness_matrix(k)
    M = np.diag(MASS)
    vals, _ = eigh(K, M)
    return np.sqrt(np.maximum(vals, 0.0))


true_freqs = natural_frequencies(TRUE_K)
obs_freqs = true_freqs[:N_MODES_OBS]   # we only observe the lowest N_MODES_OBS

print("True inter-story stiffnesses:")
print(f"  {TRUE_K}")
print(f"True frequencies (all 10 modes): {true_freqs.round(4)}")
print(f"Observed modes: f1..f{N_MODES_OBS} = {obs_freqs.round(4)}")
print()


# ---------------------------------------------------------------------------
# Prior: independent lognormal on each k_i centred on 10 N/m
# ---------------------------------------------------------------------------
def log_prior_flat(theta):
    if np.any(theta <= 0):
        return -np.inf
    # Lognormal centred on 10 N/m, sigma=0.5 in log-space (~factor-of-2 range)
    return float(np.sum(-0.5 * ((np.log(theta) - np.log(10.0)) / 0.5) ** 2))


# ---------------------------------------------------------------------------
# Likelihood factory
# ---------------------------------------------------------------------------
def make_likelihood(n_reps, seed):
    _rng = np.random.default_rng(seed)
    # Simulate n_reps measurements of each of the N_MODES_OBS frequencies
    y = np.concatenate([
        _rng.normal(obs_freqs[r], SIGMA_OBS, size=n_reps)
        for r in range(N_MODES_OBS)
    ])
    return mc.GaussianNoiseLikelihood(
        forward_model=lambda theta: np.repeat(
            natural_frequencies(theta)[:N_MODES_OBS], n_reps
        ),
        y_obs=y,
        noise_std=SIGMA_OBS,
    )


# ---------------------------------------------------------------------------
# Sampler settings
# ---------------------------------------------------------------------------
N_SAMPLES = 20_000
DISCARD = 4000
# Good initial covariance: small variance around prior centre
INIT_COV = 0.3 * np.eye(N_FLOORS)

x0 = np.full(N_FLOORS, 10.0)

param_names = [f"k{i+1}" for i in range(N_FLOORS)]


# ---------------------------------------------------------------------------
# Campaign 0: flat prior
# ---------------------------------------------------------------------------
print("=== Campaign 0: flat prior, 6 obs/mode ===")
ll0 = make_likelihood(N_REPS, seed=100)
prob0 = mc.Problem(prior=log_prior_flat, likelihood=ll0, param_names=param_names)
result0 = mc.DRAM(n_samples=N_SAMPLES, initial_cov=INIT_COV).run(prob0, x0=x0)
r0 = result0.discard(DISCARD)

print(f"  Posterior mean:  {r0.mean().round(2)}")
print(f"  Posterior std:   {r0.std().round(3)}")
print(f"  Mean abs error:  {np.abs(r0.mean() - TRUE_K).mean():.3f} N/m")
print()


# ---------------------------------------------------------------------------
# Campaign 1: posterior of 0 as Gaussian prior, 6 new obs/mode
# ---------------------------------------------------------------------------
print("=== Campaign 1: Gaussian prior from campaign 0, 6 new obs/mode ===")
prior1 = result0.as_prior(method='gaussian', discard=DISCARD)
print(f"  {prior1}")

ll1 = make_likelihood(N_REPS, seed=200)
prob1 = mc.Problem(prior=prior1, likelihood=ll1, param_names=param_names)
result1 = mc.DRAM(
    n_samples=N_SAMPLES, initial_cov=prior1.cov
).run(prob1, x0=prior1.mean)
r1 = result1.discard(DISCARD)

print(f"  Posterior mean:  {r1.mean().round(2)}")
print(f"  Posterior std:   {r1.std().round(3)}")
print(f"  Mean abs error:  {np.abs(r1.mean() - TRUE_K).mean():.3f} N/m")
print()


# ---------------------------------------------------------------------------
# Campaign 2: posterior of 1 as Gaussian prior, 10 new obs/mode
# ---------------------------------------------------------------------------
print("=== Campaign 2: Gaussian prior from campaign 1, 10 new obs/mode ===")
prior2 = result1.as_prior(method='gaussian', discard=DISCARD)
print(f"  {prior2}")

ll2 = make_likelihood(10, seed=300)
prob2 = mc.Problem(prior=prior2, likelihood=ll2, param_names=param_names)
result2 = mc.DRAM(
    n_samples=N_SAMPLES, initial_cov=prior2.cov
).run(prob2, x0=prior2.mean)
r2 = result2.discard(DISCARD)

print(f"  Posterior mean:  {r2.mean().round(2)}")
print(f"  Posterior std:   {r2.std().round(3)}")
print(f"  Mean abs error:  {np.abs(r2.mean() - TRUE_K).mean():.3f} N/m")
print()


# ---------------------------------------------------------------------------
# Batch reference: all data, flat prior
# ---------------------------------------------------------------------------
print("=== Batch reference: flat prior, all 22 obs/mode ===")
# Recreate the same three datasets combined
y_batch = np.concatenate([
    np.concatenate([
        np.concatenate([
            np.random.default_rng(s).normal(obs_freqs[r], SIGMA_OBS, size=n)
            for s, n in [(100, 6), (200, 6), (300, 10)]
        ])
        for r in range(N_MODES_OBS)
    ])
])
ll_batch = mc.GaussianNoiseLikelihood(
    forward_model=lambda theta: np.repeat(
        natural_frequencies(theta)[:N_MODES_OBS], 22
    ),
    y_obs=y_batch,
    noise_std=SIGMA_OBS,
)
prob_batch = mc.Problem(prior=log_prior_flat, likelihood=ll_batch, param_names=param_names)
result_batch = mc.DRAM(n_samples=N_SAMPLES, initial_cov=INIT_COV).run(prob_batch, x0=x0)
r_batch = result_batch.discard(DISCARD)

print(f"  Posterior mean:  {r_batch.mean().round(2)}")
print(f"  Posterior std:   {r_batch.std().round(3)}")
print(f"  Mean abs error:  {np.abs(r_batch.mean() - TRUE_K).mean():.3f} N/m")
print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("Uncertainty reduction — mean posterior std over all 10 parameters:")
print(f"  {'Stage':<45}  mean(std)")
for label, r in [
    ("Campaign 0  (flat prior,      6 obs/mode)", r0),
    ("Campaign 1  (Gaussian prior, +6 obs/mode)", r1),
    ("Campaign 2  (Gaussian prior,+10 obs/mode)", r2),
    ("Batch       (flat prior,     22 obs/mode)", r_batch),
]:
    print(f"  {label:<45}  {r.std().mean():.4f}")
print()

print("Note: KDE method is not used here because it degrades in quality and")
print("speed above ~6-8 parameters. The Gaussian approximation is appropriate")
print("for structural posteriors which are typically unimodal and near-elliptical.")
print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: posterior means vs true values, per campaign
    x_pos = np.arange(N_FLOORS)
    width = 0.22
    ax = axes[0]
    for i, (label, r, color) in enumerate([
        ("Campaign 0", r0,      "steelblue"),
        ("Campaign 1", r1,      "darkorange"),
        ("Campaign 2", r2,      "seagreen"),
        ("Batch ref",  r_batch, "gray"),
    ]):
        offset = (i - 1.5) * width
        ax.bar(x_pos + offset, r.mean(), width=width * 0.9,
               yerr=r.std(), capsize=3, label=label, color=color, alpha=0.8)
    ax.plot(x_pos, TRUE_K, 'r^', ms=8, zorder=5, label="True k")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"k{i+1}" for i in range(N_FLOORS)])
    ax.set_ylabel("Stiffness [N/m]")
    ax.set_title("Posterior mean +/- 1 std per campaign")
    ax.legend(loc="upper right", fontsize=8)

    # Bottom: std reduction over campaigns
    ax2 = axes[1]
    for i, (label, r, color) in enumerate([
        ("Campaign 0", r0,      "steelblue"),
        ("Campaign 1", r1,      "darkorange"),
        ("Campaign 2", r2,      "seagreen"),
    ]):
        offset = (i - 1) * width
        ax2.bar(x_pos + offset, r.std(), width=width * 0.9,
                label=label, color=color, alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"k{i+1}" for i in range(N_FLOORS)])
    ax2.set_ylabel("Posterior std [N/m]")
    ax2.set_title("Uncertainty shrinkage across sequential campaigns")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
