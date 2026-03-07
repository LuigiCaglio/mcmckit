"""Comparison of adaptive MCMC samplers on a 2D correlated Gaussian.

Samplers compared:
- MH       : fixed proposal (requires manual tuning)
- RAM      : adapts proposal covariance via Cholesky rank-1 update
- DRAM     : delayed rejection + adaptive covariance (AM)
- AdaptiveMALA : MALA with automatic step-size tuning
"""

import numpy as np
import mcmckit as mc

# -----------------------------------------------------------------------
# Problem
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


def grad_log_prior(theta):
    return np.zeros_like(theta)


def grad_log_likelihood(theta):
    return -true_prec @ (theta - true_mean)


problem = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["x", "y"],
    grad_log_prior=grad_log_prior,
    grad_log_likelihood=grad_log_likelihood,
)

N = 15_000
x0 = [0.0, 0.0]

# -----------------------------------------------------------------------
# Run all samplers
# -----------------------------------------------------------------------
# MH: well-tuned reference
mh = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=N)
result_mh = mh.run(problem, x0=x0)

# RAM: starts from bad initial cov, self-corrects
ram = mc.RAM(n_samples=N, initial_cov=np.eye(2) * 0.01**2)
result_ram = ram.run(problem, x0=x0)

# DRAM: adaptive cov + delayed rejection fallback
dram = mc.DRAM(n_samples=N, initial_cov=np.eye(2) * 0.5)
result_dram = dram.run(problem, x0=x0)

# AdaptiveMALA: gradient-based, auto step size
amala = mc.AdaptiveMALA(n_samples=N, initial_step_size=0.05)
result_amala = amala.run(problem, x0=x0)

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------
samplers = {
    "MH (tuned)":    result_mh,
    "RAM":           result_ram,
    "DRAM":          result_dram,
    "AdaptiveMALA":  result_amala,
}

w = 14
print("=" * (w * 5 + 4))
print(f"{'':>{w}}  {'mean[x]':>{w}}  {'mean[y]':>{w}}  {'std[x]':>{w}}  {'acc.rate':>{w}}")
print("=" * (w * 5 + 4))
for name, res in samplers.items():
    print(f"{name:>{w}}  {res.mean()[0]:>{w}.4f}  {res.mean()[1]:>{w}.4f}"
          f"  {res.std()[0]:>{w}.4f}  {res.acceptance_rate:>{w}.3f}")
print("=" * (w * 5 + 4))
print(f"{'True':>{w}}  {true_mean[0]:>{w}.4f}  {true_mean[1]:>{w}.4f}"
      f"  {np.sqrt(true_cov[0,0]):>{w}.4f}")

print(f"\nRAM final proposal std: {np.sqrt(np.diag(ram.proposal_cov))}")
print(f"DRAM final proposal std: {np.sqrt(np.diag(dram.proposal_cov))}")
print(f"AdaptiveMALA final step size: {amala.step_size:.4f}")
print(f"DRAM stage1 acc={dram.stage1_acceptance_rate:.3f}  stage2 acc={dram.stage2_acceptance_rate:.3f}")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    for name, res in samplers.items():
        res.plot_corner(true_values=true_mean, title=name)

    plt.show()

except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
