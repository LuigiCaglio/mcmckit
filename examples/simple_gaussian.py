"""Simple example: sample from a 2D correlated Gaussian posterior."""

import numpy as np
import mcmckit as mc

# True posterior: N(mu, Sigma)
true_mean = np.array([2.0, -1.0])
true_cov  = np.array([[1.0, 0.8],
                       [0.8, 1.0]])
true_prec = np.linalg.inv(true_cov)


def log_prior(theta):
    # Flat prior (returns 0 everywhere)
    return 0.0


def log_likelihood(theta):
    diff = theta - true_mean
    return -0.5 * diff @ true_prec @ diff


# Optional: analytical gradients for future gradient-based samplers (MALA, HMC).
# Standard MH ignores these — safe to omit if not needed.
def grad_log_prior(theta):
    return np.zeros_like(theta)  # flat prior -> zero gradient


def grad_log_likelihood(theta):
    return -true_prec @ (theta - true_mean)


problem = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["x", "y"],
    grad_log_prior=grad_log_prior,
    grad_log_likelihood=grad_log_likelihood,
)

# Gradients are accessible but ignored by MH:
# logp, grad = problem.log_posterior_and_grad(np.array([1.0, 0.0]))

# ------------------------------------------------------------------
# Option A: full run
# ------------------------------------------------------------------
sampler = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=10_000)
result = sampler.run(problem, x0=[0.0, 0.0])

print(result)
print("mean  :", result.mean())
print("std   :", result.std())
print(f"acceptance rate: {result.acceptance_rate:.3f}")

# ------------------------------------------------------------------
# Option B: step-by-step (inspect / stop early)
# ------------------------------------------------------------------
sampler2 = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=10_000)
sampler2.initialize(problem, x0=[0.0, 0.0])

for i in range(10_000):
    sampler2.step()
    if i % 2000 == 0:
        print(f"step {i:5d} | current = {sampler2.current} | acc = {sampler2.acceptance_rate:.3f}")

result2 = sampler2.get_result()
print("\nStep-by-step result:", result2)

# ------------------------------------------------------------------
# Option C: MALA (requires gradients, mixes faster than MH)
# ------------------------------------------------------------------
# Optimal MALA acceptance rate is ~0.57 (vs ~0.23 for MH).
# Tune step_size until you're in that range.
mala = mc.MALA(step_size=0.5, n_samples=10_000)
result_mala = mala.run(problem, x0=[0.0, 0.0])

print("\nMALA result:", result_mala)
print("MALA mean  :", result_mala.mean())
print(f"MALA acceptance rate: {result_mala.acceptance_rate:.3f}  (target ~0.57)")

# ------------------------------------------------------------------
# Accessing raw samples for custom analysis / plots
# ------------------------------------------------------------------
samples = result.samples          # np.ndarray, shape (n_samples, n_params)
log_p   = result.log_posteriors   # np.ndarray, shape (n_samples,)

# Discard burn-in and get a new Result with all methods intact
result_burned = result.discard(2_000)
print(f"full chain    : {result.samples.shape}")
print(f"post-burnin   : {result_burned.samples.shape}")
print(f"post-burnin mean: {result_burned.mean()}")

# Or access the raw array directly for custom work
samples = result_burned.samples   # plain np.ndarray, shape (8000, 2)
log_p   = result_burned.log_posteriors

# ------------------------------------------------------------------
# Plots (requires matplotlib)
# ------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    result.plot_trace(title="Trace plot")
    result.plot_marginals(title="Marginals")

    # Default corner plot (hist+KDE diagonal, 2D KDE lower triangle)
    result.plot_corner(true_values=true_mean, title="Corner plot")

    # Scatter variant
    result.plot_corner(style="scatter", true_values=true_mean, title="Corner plot – scatter")

    # Full pair plot (scatter lower, KDE upper)
    result.plot_corner(style="full", true_values=true_mean, title="Corner plot – full")

    # Pure KDE everywhere
    result.plot_corner(style="kde", true_values=true_mean, title="Corner plot – kde")

    # MALA corner plot for comparison
    result_mala.plot_corner(true_values=true_mean, title="MALA – corner plot")

    plt.show()
except ImportError:
    print("Install matplotlib for plots: pip install mcmckit[plot]")
