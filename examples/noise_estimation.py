"""Estimating measurement noise jointly with model parameters.

Three modes using GaussianNoiseLikelihood on a nonlinear forward model
  f(k, c) = k * exp(-c * x_input)   (exponential decay)

True values: k=3.0, c=0.4, sigma=0.15

Mode A — fixed sigma:     only k, c are estimated
Mode B — estimated sigma: theta = [k, c, log_sigma], all sampled jointly
Mode C — marginalised sigma: sigma is integrated out analytically via an
          Inverse-Gamma prior; only k, c are sampled. Works for any nonlinear
          forward model (no conjugacy requirement on f).
"""

import numpy as np
import mcmckit as mc

rng = np.random.default_rng(0)

# -----------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------
TRUE_K     = 3.0
TRUE_C     = 0.4
TRUE_SIGMA = 0.15
N_OBS      = 25

x_input = np.linspace(0.2, 5.0, N_OBS)
y_obs   = TRUE_K * np.exp(-TRUE_C * x_input) + rng.normal(scale=TRUE_SIGMA, size=N_OBS)


def forward_model(theta_model):
    k, c = float(theta_model[0]), float(theta_model[1])
    return k * np.exp(-c * x_input)


# -----------------------------------------------------------------------
# Shared prior on model params (k > 0, c > 0)
# -----------------------------------------------------------------------
def log_prior_model(theta_model):
    k, c = float(theta_model[0]), float(theta_model[1])
    if k <= 0 or c <= 0:
        return -np.inf
    return 0.0    # flat on (k, c) > 0


# -----------------------------------------------------------------------
# Mode A — fixed sigma
# -----------------------------------------------------------------------
ll_A = mc.GaussianNoiseLikelihood(forward_model, y_obs, noise_std=TRUE_SIGMA)
prob_A = mc.Problem(prior=log_prior_model, likelihood=ll_A,
                    param_names=["k", "c"])

res_A = mc.DRAM(n_samples=8_000, initial_cov=np.diag([0.1, 0.02])).run(
    prob_A, x0=[2.0, 0.2]).discard(2000)

# -----------------------------------------------------------------------
# Mode B — estimated sigma  (last param = log σ)
# -----------------------------------------------------------------------
def log_prior_B(theta):
    k, c, log_s = float(theta[0]), float(theta[1]), float(theta[2])
    if k <= 0 or c <= 0:
        return -np.inf
    sigma = np.exp(log_s)
    return -0.5 * sigma**2    # HalfNormal(1) on sigma

ll_B = mc.GaussianNoiseLikelihood(forward_model, y_obs)   # noise_std=None
prob_B = mc.Problem(prior=log_prior_B, likelihood=ll_B,
                    param_names=["k", "c", "log_sigma"])

res_B = mc.DRAM(n_samples=8_000, initial_cov=np.diag([0.1, 0.02, 0.1])).run(
    prob_B, x0=[2.0, 0.2, np.log(0.3)]).discard(2000)

sigma_post_B = np.exp(res_B.samples[:, 2])

# -----------------------------------------------------------------------
# Mode C — marginalised sigma  (InvGamma prior on sigma^2)
# -----------------------------------------------------------------------
ll_C = mc.GaussianNoiseLikelihood(forward_model, y_obs,
                                  marginalise_noise=True,
                                  inv_gamma_alpha=1.0, inv_gamma_beta=1e-4)
prob_C = mc.Problem(prior=log_prior_model, likelihood=ll_C,
                    param_names=["k", "c"])

res_C = mc.DRAM(n_samples=8_000, initial_cov=np.diag([0.1, 0.02])).run(
    prob_C, x0=[2.0, 0.2]).discard(2000)

# Posterior point estimate of sigma via posterior mean of InvGamma
sigma_post_C_mean = np.mean([ll_C.posterior_sigma(s) for s in res_C.samples])

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"{'':30s}  {'k':>10}  {'c':>10}  {'sigma':>10}")
print("-" * 65)
print(f"{'True':30s}  {TRUE_K:10.4f}  {TRUE_C:10.4f}  {TRUE_SIGMA:10.4f}")
print(f"{'A — fixed sigma':30s}  {res_A.mean()[0]:10.4f}  {res_A.mean()[1]:10.4f}  {'(fixed)':>10}")
print(f"{'B — estimated sigma':30s}  {res_B.mean()[0]:10.4f}  {res_B.mean()[1]:10.4f}  {sigma_post_B.mean():10.4f}")
print(f"{'C — marginalised sigma':30s}  {res_C.mean()[0]:10.4f}  {res_C.mean()[1]:10.4f}  {sigma_post_C_mean:10.4f}")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    res_A.plot_corner(true_values=[TRUE_K, TRUE_C],
                      title="Mode A: fixed sigma")
    res_B.plot_corner(true_values=[TRUE_K, TRUE_C, np.log(TRUE_SIGMA)],
                      title="Mode B: estimated sigma  (last param = log σ)")
    res_C.plot_corner(true_values=[TRUE_K, TRUE_C],
                      title="Mode C: marginalised sigma")

    # Posterior predictive for Mode C
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x_fine = np.linspace(x_input.min(), x_input.max(), 200)
    for s in rng.choice(len(res_C.samples), size=150, replace=False):
        k, c = res_C.samples[s]
        ax.plot(x_fine, k * np.exp(-c * x_fine),
                color="steelblue", alpha=0.07, lw=0.8)
    ax.plot(x_fine, TRUE_K * np.exp(-TRUE_C * x_fine),
            color="crimson", lw=1.5, label=f"true (k={TRUE_K}, c={TRUE_C})")
    ax.scatter(x_input, y_obs, s=20, color="k", zorder=5, label="observations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior predictive — Mode C (marginalised σ)")
    ax.legend()

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
