"""Hierarchical Bayesian model updating across multiple structures.

Scenario: 5 nominally identical shear frames, each with the same design
stiffness but manufactured with slight variations.  Each structure is
monitored independently, but by pooling information through shared
hyperparameters we get better estimates — especially for structures with
few measurements.

Model
-----
  phi = [mu_k, log_sigma_k]           hyperparameters
  k_j | phi ~ Normal(mu_k, sigma_k^2) group-level prior
  y_j | k_j ~ Normal(omega(k_j), sigma_obs^2)  likelihood

where omega(k) = sqrt(k / m) is the natural frequency of a SDOF oscillator.

True values:  mu_k = 10.0,  sigma_k = 0.8 (population spread),
              sigma_obs = 0.05 (measurement noise, fixed)

The pooling effect is demonstrated by comparing:
  - Independent per-structure estimates (no pooling)
  - Hierarchical estimates (full pooling)
"""

import numpy as np
import mcmckit as mc

rng = np.random.default_rng(42)

# -----------------------------------------------------------------------
# System parameters
# -----------------------------------------------------------------------
MASS         = 1.0
TRUE_MU_K    = 10.0
TRUE_SIGMA_K = 0.8
SIGMA_OBS    = 0.05
J            = 5        # number of structures
N_OBS        = [6, 6, 3, 3, 2]  # few obs for structures 3-5 to show pooling

# -----------------------------------------------------------------------
# Simulate true stiffnesses and observations
# -----------------------------------------------------------------------
true_k = rng.normal(TRUE_MU_K, TRUE_SIGMA_K, size=J)

def omega(k):
    return np.sqrt(max(k, 1e-6) / MASS)

datasets = [
    rng.normal(omega(true_k[j]), SIGMA_OBS, size=N_OBS[j])
    for j in range(J)
]

print("True stiffness per structure:", np.round(true_k, 3))
print(f"True population: mu={TRUE_MU_K}, sigma_k={TRUE_SIGMA_K}")
print()

# -----------------------------------------------------------------------
# Build HierarchicalProblem
# -----------------------------------------------------------------------

def hyperprior(phi):
    """p(mu_k, log_sigma_k): weakly informative Normal × HalfNormal."""
    mu_k, log_sigma_k = phi
    sigma_k = np.exp(log_sigma_k)
    return (-0.5 * ((mu_k - 10.0) / 5.0)**2    # Normal(10, 5) on mu_k
            - 0.5 * sigma_k**2)                 # HalfNormal(1) on sigma_k


def group_prior(theta_j, phi):
    """p(k_j | mu_k, sigma_k): Normal shrinkage prior."""
    k_j = float(theta_j[0])
    if k_j <= 0:
        return -np.inf
    mu_k, log_sigma_k = phi
    sigma_k = np.exp(log_sigma_k)
    return -0.5 * ((k_j - mu_k) / sigma_k)**2 - log_sigma_k


def make_likelihood(y_j):
    """Fixed-noise Gaussian likelihood for structure j."""
    ll = mc.GaussianNoiseLikelihood(
        forward_model=lambda theta: np.full(len(y_j), omega(float(theta[0]))),
        y_obs=y_j,
        noise_std=SIGMA_OBS,
    )
    return ll


likelihoods = [make_likelihood(datasets[j]) for j in range(J)]

hproblem = mc.HierarchicalProblem(
    hyperprior=hyperprior,
    group_prior=group_prior,
    group_likelihoods=likelihoods,
    n_hyper=2,
    n_group=1,
    param_names_hyper=["mu_k", "log_sigma_k"],
    param_names_group=["k"],
)

print(f"Total parameters: {hproblem.n_params}  "
      f"(2 hyper + {J} x 1 group)")

# -----------------------------------------------------------------------
# Initial point and proposal covariance
# -----------------------------------------------------------------------
x0 = hproblem.default_x0(
    phi0=[10.0, np.log(1.0)],
    group_x0s=[[10.0]] * J,
)
init_cov = np.diag([0.5, 0.1] + [0.3] * J)

# -----------------------------------------------------------------------
# Run DRAM on the joint parameter vector
# -----------------------------------------------------------------------
sampler = mc.DRAM(n_samples=25_000, initial_cov=init_cov)
result  = sampler.run(hproblem, x0=x0)
result  = result.discard(5000)

# -----------------------------------------------------------------------
# Extract marginal posteriors
# -----------------------------------------------------------------------
hyper_result = hproblem.extract_hyper(result)
group_results = [hproblem.extract_group(result, j) for j in range(J)]

sigma_k_samples = np.exp(hyper_result.samples[:, 1])

print("=== Hyperparameter posterior ===")
print(f"  mu_k      : {hyper_result.mean()[0]:.3f} +/- {hyper_result.std()[0]:.3f}"
      f"  (true: {TRUE_MU_K})")
print(f"  sigma_k   : {sigma_k_samples.mean():.3f} +/- {sigma_k_samples.std():.3f}"
      f"  (true: {TRUE_SIGMA_K})")
print()

print(f"{'j':>3}  {'n_obs':>5}  {'true k':>8}  {'hier mean':>10}  {'hier std':>9}  {'indep mean':>11}  {'indep std':>10}")
print("-" * 65)

# -----------------------------------------------------------------------
# Independent estimates for comparison (no pooling)
# -----------------------------------------------------------------------
indep_means = []
indep_stds  = []

for j in range(J):
    # Flat prior, fixed noise — same likelihood, no hyperprior
    def log_prior_flat(theta, j=j):
        return 0.0 if float(theta[0]) > 0 else -np.inf

    prob_j = mc.Problem(
        prior=log_prior_flat,
        likelihood=likelihoods[j],
        param_names=[f"k_{j}"],
    )
    res_j = mc.DRAM(n_samples=8000, initial_cov=np.array([[0.5]])).run(
        prob_j, x0=[10.0]).discard(2000)
    indep_means.append(res_j.mean()[0])
    indep_stds.append(res_j.std()[0])

for j in range(J):
    hm = group_results[j].mean()[0]
    hs = group_results[j].std()[0]
    print(f"{j:>3}  {N_OBS[j]:>5}  {true_k[j]:>8.3f}"
          f"  {hm:>10.3f}  {hs:>9.3f}"
          f"  {indep_means[j]:>11.3f}  {indep_stds[j]:>10.3f}")

# -----------------------------------------------------------------------
# Posterior predictive: propagate to natural frequency
# -----------------------------------------------------------------------
print()
print("Posterior predictive check (structure 0):")
pp = group_results[0].posterior_predictive(
    forward_model=lambda theta: [omega(theta[0])],
    n_eval=500,
)
print(f"  predicted omega: {pp.mean()[0]:.4f} +/- {pp.std()[0]:.4f}  "
      f"(true: {omega(true_k[0]):.4f})")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    # Hyperparameter posteriors
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].hist(hyper_result.samples[:, 0], bins=50, density=True,
                 color="steelblue", alpha=0.7)
    axes[0].axvline(TRUE_MU_K, color="crimson", ls="--", lw=1.5, label="true")
    axes[0].set_xlabel("mu_k")
    axes[0].set_ylabel("density")
    axes[0].legend()

    axes[1].hist(sigma_k_samples, bins=50, density=True,
                 color="steelblue", alpha=0.7)
    axes[1].axvline(TRUE_SIGMA_K, color="crimson", ls="--", lw=1.5, label="true")
    axes[1].set_xlabel("sigma_k")
    axes[1].legend()
    fig.suptitle("Hyperparameter posteriors")

    # Per-structure: hierarchical vs independent
    fig2, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x_pos = np.arange(J)
    h_means = np.array([group_results[j].mean()[0] for j in range(J)])
    h_stds  = np.array([group_results[j].std()[0]  for j in range(J)])
    ax.errorbar(x_pos - 0.15, h_means, yerr=2*h_stds, fmt="o",
                color="steelblue", label="hierarchical (±2σ)", capsize=4)
    ax.errorbar(x_pos + 0.15, indep_means, yerr=2*np.array(indep_stds),
                fmt="s", color="darkorange", label="independent (±2σ)", capsize=4)
    ax.scatter(x_pos, true_k, marker="*", s=120, color="crimson",
               zorder=5, label="true k")
    ax.axhline(TRUE_MU_K, color="grey", ls=":", lw=1, label="population mean")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"struct {j}\n(n={N_OBS[j]})" for j in range(J)])
    ax.set_ylabel("stiffness k")
    ax.legend(fontsize=8)
    ax.set_title("Hierarchical vs independent estimates (pooling effect)")

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
