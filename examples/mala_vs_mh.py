"""MALA vs Metropolis-Hastings on a 2D correlated Gaussian.

Shows:
- How to provide gradients to Problem
- MALA usage (identical interface to MH)
- Side-by-side comparison of trace plots and corner plots
- How acceptance rate and mixing differ between the two samplers
"""

import numpy as np
import mcmckit as mc

# -----------------------------------------------------------------------
# Problem: 2D correlated Gaussian posterior
# -----------------------------------------------------------------------
true_mean = np.array([2.0, -1.0])
true_cov  = np.array([[1.0, 0.8],
                       [0.8, 1.0]])
true_prec = np.linalg.inv(true_cov)


def log_prior(theta):
    return 0.0                              # flat prior


def log_likelihood(theta):
    diff = theta - true_mean
    return -0.5 * diff @ true_prec @ diff


def grad_log_prior(theta):
    return np.zeros_like(theta)             # flat prior -> zero gradient


def grad_log_likelihood(theta):
    return -true_prec @ (theta - true_mean)


problem = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["x", "y"],
    grad_log_prior=grad_log_prior,
    grad_log_likelihood=grad_log_likelihood,
)

# -----------------------------------------------------------------------
# Run both samplers from the same starting point
# -----------------------------------------------------------------------
N = 10_000
x0 = [0.0, 0.0]

mh   = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=N)
mala = mc.MALA(step_size=0.5, n_samples=N)

result_mh   = mh.run(problem, x0=x0)
result_mala = mala.run(problem, x0=x0)

# -----------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------
print("=" * 45)
print(f"{'':15s}  {'MH':>12s}  {'MALA':>12s}")
print("=" * 45)
print(f"{'mean[x]':15s}  {result_mh.mean()[0]:>12.4f}  {result_mala.mean()[0]:>12.4f}")
print(f"{'mean[y]':15s}  {result_mh.mean()[1]:>12.4f}  {result_mala.mean()[1]:>12.4f}")
print(f"{'std[x]':15s}  {result_mh.std()[0]:>12.4f}  {result_mala.std()[0]:>12.4f}")
print(f"{'std[y]':15s}  {result_mh.std()[1]:>12.4f}  {result_mala.std()[1]:>12.4f}")
print(f"{'accept. rate':15s}  {result_mh.acceptance_rate:>12.3f}  {result_mala.acceptance_rate:>12.3f}")
print("=" * 45)
print(f"True mean : {true_mean}")
print(f"True std  : {np.sqrt(np.diag(true_cov))}")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    # --- Trace plots side by side ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharey="row")
    fig.suptitle("Trace plots: MH (left) vs MALA (right)")

    for i, name in enumerate(["x", "y"]):
        axes[i, 0].plot(result_mh.samples[:, i], lw=0.5, color="steelblue")
        axes[i, 0].axhline(true_mean[i], color="crimson", ls="--", lw=1)
        axes[i, 0].set_ylabel(name)
        axes[i, 0].set_title(f"MH  (acc={result_mh.acceptance_rate:.2f})" if i == 0 else "")

        axes[i, 1].plot(result_mala.samples[:, i], lw=0.5, color="darkorange")
        axes[i, 1].axhline(true_mean[i], color="crimson", ls="--", lw=1)
        axes[i, 1].set_title(f"MALA  (acc={result_mala.acceptance_rate:.2f})" if i == 0 else "")

    for ax in axes[-1]:
        ax.set_xlabel("iteration")
    fig.tight_layout()

    # --- Corner plots side by side ---
    result_mh.plot_corner(true_values=true_mean,
                          title=f"MH corner  (acc={result_mh.acceptance_rate:.2f})")

    result_mala.plot_corner(true_values=true_mean,
                            title=f"MALA corner  (acc={result_mala.acceptance_rate:.2f})")

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
