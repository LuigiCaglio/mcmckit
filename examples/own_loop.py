"""You own the loop: single-step sampling.

This is the interface mcmckit is built around. Each sampler is a plain
function that advances the chain by one step; you write the recursion.

Demonstrated here, on a 3-storey shear building whose lower two storeys are
identified from its natural frequencies:

  1. a hand-written RAM loop
  2. keeping the forward model's output (frequencies) for every sample
  3. stopping early on your own criterion
  4. the same thing again via the full-run helper, for comparison

Run:  python examples/own_loop.py
"""

import numpy as np

import mcmckit as mc


# ======================================================================
# A forward model: natural frequencies of a 3-storey shear building
# ======================================================================

M_STOREY = 1.0e5          # kg per floor
K_NOMINAL = 1.6e8         # N/m per storey
N_DOF = 3

# The top storey is known (inspected, or instrumented). Identifying all three
# storeys from three frequencies alone is not identifiable: several stiffness
# combinations reproduce the same spectrum to within measurement noise.
K3_KNOWN = 1.0


def natural_frequencies(stiffness_factors):
    """Return the 3 natural frequencies [Hz] for given stiffness factors."""
    k = K_NOMINAL * np.asarray(stiffness_factors, dtype=float)

    K = np.zeros((N_DOF, N_DOF))
    for i in range(N_DOF):
        K[i, i] += k[i]
        if i + 1 < N_DOF:
            K[i, i + 1] -= k[i + 1]
            K[i + 1, i] -= k[i + 1]
            K[i + 1, i + 1] += k[i + 1]
    M = np.eye(N_DOF) * M_STOREY

    eigenvalues = np.linalg.eigvalsh(np.linalg.solve(M, K))
    return np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)


# ---- synthetic measurement: storey 2 has lost 30% of its stiffness ----
TRUE_PARAMS = np.array([1.0, 0.7])                 # k1, k2
SIGMA = 0.02                                       # Hz, measurement noise
rng_data = np.random.default_rng(0)
MEASURED = (natural_frequencies([*TRUE_PARAMS, K3_KNOWN])
            + rng_data.normal(0, SIGMA, N_DOF))


def log_post(theta):
    """Log posterior, returning the frequencies alongside it.

    Returning a tuple opts into auxiliary passthrough: mcmckit threads the
    second element back out with the accepted sample, so the frequencies are
    kept without ever re-running the model.
    """
    if np.any(theta <= 0.1) or np.any(theta > 2.0):     # uniform prior box
        return -np.inf, np.full(N_DOF, np.nan)

    freqs = natural_frequencies([*theta, K3_KNOWN])
    log_lik = -0.5 * np.sum(((freqs - MEASURED) / SIGMA) ** 2)
    return log_lik, freqs


# ======================================================================
# 1. A hand-written RAM loop
# ======================================================================

def main():
    n_par = len(TRUE_PARAMS)
    print(f"true stiffness factors : {TRUE_PARAMS}  (k3 known = {K3_KNOWN})")
    print(f"measured frequencies   : {np.round(MEASURED, 3)} Hz")
    print()

    n_iter = 20_000
    burn_in = 4_000

    # ---- state, all of it explicit -----------------------------------
    x = np.ones(n_par)                                   # start at nominal
    logp, freqs = log_post(x)
    S = np.linalg.cholesky(np.eye(n_par) * 0.05**2)      # RAM adaptation state
    rng = np.random.default_rng(42)

    # ---- storage is yours too ----------------------------------------
    chain = np.zeros((n_iter, n_par))
    freq_history = np.zeros((n_iter, N_DOF))
    n_accepted = 0

    for i in range(1, n_iter + 1):
        x, logp, S, accepted, freqs = mc.ram_step(
            log_post, x, logp, S, i, aux=freqs, rng=rng
        )

        chain[i - 1] = x
        freq_history[i - 1] = freqs          # kept for free, no extra model calls
        n_accepted += accepted

        # ---- 3. your own stopping rule, checked whenever you like ----
        if i % 5_000 == 0:
            recent = chain[max(0, i - 5_000):i]
            print(f"  step {i:>6}  acc={n_accepted / i:.2f}  "
                  f"mean={np.round(recent.mean(0), 3)}  "
                  f"scale={np.round(np.sqrt(np.diag(S @ S.T)), 4)}")

            if i >= 15_000 and recent.std(0).max() < 1e-4:
                print("  converged by my own criterion, stopping early")
                chain = chain[:i]
                freq_history = freq_history[:i]
                break

    posterior = chain[burn_in:]
    print()
    print(f"posterior mean : {np.round(posterior.mean(0), 3)}   (true {TRUE_PARAMS})")
    print(f"posterior std  : {np.round(posterior.std(0), 3)}")
    print(f"acceptance     : {n_accepted / len(chain):.2f}")
    print()

    # The frequencies came along for the ride, so the posterior predictive
    # costs nothing extra.
    pred = freq_history[burn_in:]
    print("posterior predictive frequencies [Hz]")
    for j in range(N_DOF):
        print(f"  mode {j + 1}: {pred[:, j].mean():6.3f} "
              f"+/- {pred[:, j].std():.3f}   (measured {MEASURED[j]:6.3f})")
    print()

    # ==================================================================
    # 4. The same run, handing the loop over
    # ==================================================================
    print("same problem via the full-run helper:")
    result = mc.ram(
        lambda th: log_post(th)[0],          # helper wants just the density
        x0=np.ones(n_par),
        n_samples=n_iter,
        initial_cov=0.05**2,
        param_names=["k1", "k2"],
        rng=np.random.default_rng(42),
    )
    trimmed = result.discard(burn_in)
    print(f"  posterior mean : {np.round(trimmed.mean(), 3)}")
    print(f"  acceptance     : {result.acceptance_rate:.2f}")

    # ---- optional plots ---------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to see the plots)")
        return

    fig, axes = plt.subplots(n_par, 1, figsize=(9, 5), sharex=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(chain[:, j], lw=0.5)
        ax.axhline(TRUE_PARAMS[j], color="crimson", ls="--", label="true")
        ax.set_ylabel(f"$k_{j + 1}$")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("iteration")
    fig.suptitle("RAM, hand-written loop")
    fig.tight_layout()

    # A chain you built yourself wraps straight into Result for the plots.
    mc.Result(samples=posterior, param_names=["k1", "k2"]).plot_corner(
        true_values=TRUE_PARAMS, title="Posterior"
    )
    plt.show()


if __name__ == "__main__":
    main()
