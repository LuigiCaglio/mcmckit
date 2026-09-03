"""Model updating with a black-box OpenSeesPy forward model, in parallel.

Identifies two storey-stiffness scale factors of a 6-storey shear building from
its modal frequencies. The likelihood builds and solves a real OpenSees model on
every call, which is the normal situation in structural model updating: the
forward model is a black box that mcmckit only ever calls.

Two things make this work, and both matter:

1. **The likelihood is a module-level function.** Worker processes receive it by
   pickling, which a lambda or a closure cannot survive. Data the likelihood
   needs (here ``F_OBS``) lives at module level too, so each worker gets it on
   import.

2. **Processes, never threads.** OpenSeesPy keeps a *single global model
   domain*. Two threads calling ``ops.model`` at once overwrite each other -
   measured, that gives an ``OpenSeesError`` on a good day and a segmentation
   fault on a bad one. Separate processes each get their own domain, which is
   why ``backend="process"`` is correct here and ``backend="thread"`` is not.
   mcmckit's default ``backend="auto"`` picks processes, and refuses to run
   rather than silently falling back to threads.

Run it with::

    python openseespy_parallel.py

Requires ``openseespy``.
"""

import time

import numpy as np
import openseespy.opensees as ops

import mcmckit as mc

# ---------------------------------------------------------------------------
# The structure
# ---------------------------------------------------------------------------

N_STORY = 6
N_MODES = 3
M_STORY = 1000.0            # kg per floor
K_NOMINAL = 4.0e6           # N/m per storey
TRUE_SCALE = np.array([0.85, 1.00])     # 15% stiffness loss in the lower half
NOISE_LEVEL = 0.01                      # 1% scatter on the measured frequencies


def build_and_eigen(scale):
    """Build the shear building in OpenSees and return its first frequencies (Hz).

    This is the black box. ``ops.wipe()`` at both ends matters: OpenSees keeps
    one global domain, so the model must be torn down and rebuilt every call.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    ops.node(0, 0.0)
    ops.fix(0, 1)
    for i in range(1, N_STORY + 1):
        ops.node(i, 0.0)                 # coincident nodes joined by zeroLength
        ops.mass(i, M_STORY)

    for i in range(1, N_STORY + 1):
        half = 0 if i <= N_STORY // 2 else 1
        ops.uniaxialMaterial("Elastic", i, K_NOMINAL * float(scale[half]))
        ops.element("zeroLength", i, i - 1, i, "-mat", i, "-dir", 1)

    eigenvalues = ops.eigen("-fullGenLapack", N_MODES)
    ops.wipe()
    return np.sqrt(np.abs(np.array(eigenvalues))) / (2 * np.pi)


# The "measurement". Module level, so every worker has it after importing.
F_OBS = build_and_eigen(TRUE_SCALE)


# ---------------------------------------------------------------------------
# The Bayesian problem - both functions at module level so they pickle
# ---------------------------------------------------------------------------

def log_prior(theta):
    """Uniform on [0.3, 1.7] for each stiffness scale factor."""
    theta = np.asarray(theta, dtype=float)
    if np.any(theta < 0.3) or np.any(theta > 1.7):
        return -np.inf
    return 0.0


def log_likelihood(theta):
    """Gaussian on the relative frequency error."""
    frequencies = build_and_eigen(np.asarray(theta, dtype=float))
    residual = (frequencies - F_OBS) / (NOISE_LEVEL * F_OBS)
    return float(-0.5 * np.sum(residual**2))


PROBLEM = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["k_lower", "k_upper"],
)


def main():
    print(f"true scale factors : {TRUE_SCALE}")
    print(f"measured frequencies: {np.round(F_OBS, 3)} Hz\n")

    n_particles = 400
    rng = np.random.default_rng(0)
    prior_samples = rng.uniform(0.3, 1.7, size=(n_particles, 2))

    for n_workers in (1, 4):
        # Seed before each run: TMCMC draws its proposals from the global NumPy
        # stream in the parent process and only farms out likelihood calls, so
        # with the same seed the two runs are bit-identical. Any difference here
        # would mean parallelism had changed the answer.
        np.random.seed(42)
        start = time.perf_counter()
        result = mc.TMCMC(
            n_particles=n_particles,
            n_mcmc_steps=3,
            n_workers=n_workers,
            backend="process",       # never "thread" with OpenSeesPy
        ).run(PROBLEM, prior_samples=prior_samples)
        elapsed = time.perf_counter() - start

        mean, std = result.mean(), result.std()
        print(
            f"n_workers={n_workers}: {elapsed:5.1f}s  "
            f"k_lower={mean[0]:.3f}+/-{std[0]:.3f}  "
            f"k_upper={mean[1]:.3f}+/-{std[1]:.3f}  "
            f"logZ={result.log_evidence:.3f}"
        )

    print(
        "\nThe estimates are identical regardless of worker count: parallelism "
        "changes only how the work is scheduled.\n"
        "\nNote that 4 workers is *slower* here. This model solves in about two "
        "milliseconds, so handing it to another process costs more than the "
        "solve itself. Workers pay off once a single forward model takes tens of "
        "milliseconds or more, which is the usual case for a real structure - a "
        "nonlinear time history rather than six eigenvalues."
    )


if __name__ == "__main__":
    # Required. Worker processes import this module, and without the guard they
    # would re-run the whole script.
    main()
