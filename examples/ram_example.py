"""RAM (Robust Adaptive Metropolis) example.

Demonstrates:
- RAM usage: no proposal tuning needed beyond an initial covariance guess
- How the proposal covariance adapts during sampling
- Comparison with fixed-proposal MH
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
    return 0.0


def log_likelihood(theta):
    diff = theta - true_mean
    return -0.5 * diff @ true_prec @ diff


problem = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["x", "y"],
)

N = 20_000
x0 = [0.0, 0.0]

# -----------------------------------------------------------------------
# RAM: intentionally bad initial covariance — it self-corrects
# -----------------------------------------------------------------------
# Starting with identity * 0.001² (way too small for this posterior).
# RAM will automatically scale it up toward the right size.
ram = mc.RAM(n_samples=N, initial_cov=np.eye(2) * 0.001**2)
result_ram = ram.run(problem, x0=x0)

# -----------------------------------------------------------------------
# MH with a well-tuned proposal for reference
# -----------------------------------------------------------------------
mh = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=N)
result_mh = mh.run(problem, x0=x0)

# -----------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------
print("=" * 50)
print(f"{'':18s}  {'MH (tuned)':>12s}  {'RAM':>12s}")
print("=" * 50)
print(f"{'mean[x]':18s}  {result_mh.mean()[0]:>12.4f}  {result_ram.mean()[0]:>12.4f}")
print(f"{'mean[y]':18s}  {result_mh.mean()[1]:>12.4f}  {result_ram.mean()[1]:>12.4f}")
print(f"{'std[x]':18s}  {result_mh.std()[0]:>12.4f}  {result_ram.std()[0]:>12.4f}")
print(f"{'std[y]':18s}  {result_mh.std()[1]:>12.4f}  {result_ram.std()[1]:>12.4f}")
print(f"{'acceptance rate':18s}  {result_mh.acceptance_rate:>12.3f}  {result_ram.acceptance_rate:>12.3f}")
print("=" * 50)
print(f"True mean : {true_mean}")
print(f"True std  : {np.sqrt(np.diag(true_cov))}")
print()
print("RAM final proposal covariance:")
print(ram.proposal_cov)
print()
print("True posterior covariance:")
print(true_cov)

# -----------------------------------------------------------------------
# Step-by-step: watch the covariance adapt in real time
# -----------------------------------------------------------------------
print("\n--- Covariance adaptation during sampling ---")
ram2 = mc.RAM(n_samples=N, initial_cov=np.eye(2) * 0.001**2)
ram2.initialize(problem, x0=x0)

checkpoints = {1, 100, 500, 2000, 5000, 10000, N}
for i in range(1, N + 1):
    ram2.step()
    if i in checkpoints:
        cov = ram2.proposal_cov
        print(f"  step {i:>6d} | acc={ram2.acceptance_rate:.3f} "
              f"| proposal std: [{cov[0,0]**0.5:.4f}, {cov[1,1]**0.5:.4f}]"
              f"| corr: {cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]):+.3f}")

result_ram2 = ram2.get_result()

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    # Trace comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharey="row")
    fig.suptitle("Trace plots: MH tuned (left) vs RAM self-tuned (right)")
    for i, name in enumerate(["x", "y"]):
        axes[i, 0].plot(result_mh.samples[:, i], lw=0.4, color="steelblue")
        axes[i, 0].axhline(true_mean[i], color="crimson", ls="--", lw=1)
        axes[i, 0].set_ylabel(name)
        axes[i, 0].set_title(f"MH  (acc={result_mh.acceptance_rate:.2f})" if i == 0 else "")

        axes[i, 1].plot(result_ram.samples[:, i], lw=0.4, color="darkorange")
        axes[i, 1].axhline(true_mean[i], color="crimson", ls="--", lw=1)
        axes[i, 1].set_title(f"RAM  (acc={result_ram.acceptance_rate:.2f})" if i == 0 else "")
    for ax in axes[-1]:
        ax.set_xlabel("iteration")

    # Corner plots
    result_mh.plot_corner(true_values=true_mean,
                          title=f"MH (tuned)  acc={result_mh.acceptance_rate:.2f}")
    result_ram.plot_corner(true_values=true_mean,
                           title=f"RAM (self-tuned from tiny initial cov)  acc={result_ram.acceptance_rate:.2f}")

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
