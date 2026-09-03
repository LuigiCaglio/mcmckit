"""Correctness tests: do the samplers actually target the right distribution?

`test_samplers.py` checks that things run and have the right shape. Those tests
pass even if a sampler draws from the wrong distribution. These check the
numbers, against targets whose answers are known in closed form:

- every sampler must recover the mean and covariance of a correlated Gaussian
- TMCMC must recover the analytic log-evidence of a conjugate model
- the diagnostics must call converged chains converged and stuck chains stuck

Tolerances are loose relative to the Monte Carlo error actually observed (the
worst sampler is off by ~0.1 in the mean, and the limits here are 0.3). They are
set to catch a sampler that is wrong, not to pin down its efficiency.
"""

import numpy as np
import pytest

import mcmckit as mc

# ---------------------------------------------------------------------------
# Target: correlated 2-D Gaussian with a known mean and covariance
# ---------------------------------------------------------------------------

MU = np.array([1.0, -2.0])
SIGMA = np.array([[2.0, 0.8], [0.8, 1.0]])
SIGMA_INV = np.linalg.inv(SIGMA)


def _log_likelihood(theta):
    d = np.asarray(theta, dtype=float) - MU
    return -0.5 * float(d @ SIGMA_INV @ d)


def _grad_log_likelihood(theta):
    d = np.asarray(theta, dtype=float) - MU
    return -SIGMA_INV @ d


@pytest.fixture
def gaussian_problem():
    return mc.Problem(
        prior=lambda theta: 0.0,
        likelihood=_log_likelihood,
        param_names=["a", "b"],
        grad_log_likelihood=_grad_log_likelihood,
        grad_log_prior=lambda theta: np.zeros(2),
    )


# (name, factory, n_samples, burn_in) - DRAM gets fewer samples because its
# delayed-rejection stage makes it much slower per sample.
SAMPLERS = [
    ("MetropolisHastings", lambda n: mc.MetropolisHastings(proposal_cov=SIGMA * 1.2, n_samples=n), 20_000, 4_000),
    ("MALA", lambda n: mc.MALA(step_size=0.35, n_samples=n), 20_000, 4_000),
    ("AdaptiveMALA", lambda n: mc.AdaptiveMALA(n_samples=n, initial_step_size=0.3), 20_000, 4_000),
    ("RAM", lambda n: mc.RAM(n_samples=n), 20_000, 4_000),
    ("DRAM", lambda n: mc.DRAM(n_samples=n), 8_000, 2_000),
    ("Gibbs", lambda n: mc.Gibbs(n_samples=n, proposal_std=1.0), 20_000, 4_000),
]


@pytest.mark.parametrize(
    "name,factory,n_samples,burn_in", SAMPLERS, ids=[s[0] for s in SAMPLERS]
)
def test_sampler_recovers_the_target_moments(
    gaussian_problem, name, factory, n_samples, burn_in
):
    """The whole point of a sampler: its samples must have the target's moments."""
    np.random.seed(0)
    result = factory(n_samples).run(gaussian_problem, x0=[0.0, 0.0]).discard(burn_in)

    mean_err = np.abs(result.mean() - MU).max()
    assert mean_err < 0.3, f"{name} mean off by {mean_err:.3f}: {result.mean()} vs {MU}"

    cov_err = np.abs(result.cov() - SIGMA).max()
    assert cov_err < 0.5, f"{name} covariance off by {cov_err:.3f}:\n{result.cov()}\nvs\n{SIGMA}"


@pytest.mark.parametrize(
    "name,factory,n_samples,burn_in", SAMPLERS, ids=[s[0] for s in SAMPLERS]
)
def test_sampler_gets_the_correlation_sign_right(
    gaussian_problem, name, factory, n_samples, burn_in
):
    """A sampler ignoring the off-diagonal would still pass a marginal-only check."""
    np.random.seed(1)
    result = factory(n_samples).run(gaussian_problem, x0=[0.0, 0.0]).discard(burn_in)
    corr = result.cov()[0, 1] / np.sqrt(result.cov()[0, 0] * result.cov()[1, 1])
    expected = SIGMA[0, 1] / np.sqrt(SIGMA[0, 0] * SIGMA[1, 1])
    assert abs(corr - expected) < 0.15, f"{name} correlation {corr:.3f} vs {expected:.3f}"


def test_samplers_are_reproducible_under_a_fixed_seed(gaussian_problem):
    """Randomness comes from the global NumPy stream, so seeding must pin results."""
    def run():
        np.random.seed(12345)
        return mc.MetropolisHastings(proposal_cov=SIGMA, n_samples=500).run(
            gaussian_problem, x0=[0.0, 0.0]
        ).samples

    assert np.array_equal(run(), run())


# ---------------------------------------------------------------------------
# TMCMC log-evidence against a conjugate model
# ---------------------------------------------------------------------------

TAU, SIGMA_OBS = 3.0, 1.0          # prior sd, observation noise sd
Y_OBS = np.array([1.0, -0.5])


def _analytic_log_evidence():
    """theta ~ N(0, tau^2 I), y | theta ~ N(theta, sigma^2 I)  =>  y ~ N(0, (tau^2+sigma^2) I)."""
    v = TAU**2 + SIGMA_OBS**2
    return float(np.sum(-0.5 * np.log(2 * np.pi * v) - Y_OBS**2 / (2 * v)))


@pytest.fixture
def conjugate_problem():
    def log_prior(theta):
        theta = np.asarray(theta, dtype=float)
        return float(np.sum(-0.5 * np.log(2 * np.pi * TAU**2) - theta**2 / (2 * TAU**2)))

    def log_likelihood(theta):
        theta = np.asarray(theta, dtype=float)
        return float(
            np.sum(
                -0.5 * np.log(2 * np.pi * SIGMA_OBS**2)
                - (Y_OBS - theta) ** 2 / (2 * SIGMA_OBS**2)
            )
        )

    return mc.Problem(prior=log_prior, likelihood=log_likelihood, param_names=["a", "b"])


def test_tmcmc_recovers_the_analytic_log_evidence(conjugate_problem):
    """The log-evidence is what every Bayes factor rests on, so pin it to the truth."""
    np.random.seed(0)
    prior_samples = np.random.normal(0.0, TAU, size=(1000, 2))
    result = mc.TMCMC(n_particles=1000, n_mcmc_steps=5).run(
        conjugate_problem, prior_samples=prior_samples
    )
    expected = _analytic_log_evidence()
    assert abs(result.log_evidence - expected) < 0.3, (
        f"log-evidence {result.log_evidence:.4f} vs analytic {expected:.4f}"
    )


def test_tmcmc_recovers_the_conjugate_posterior_mean(conjugate_problem):
    """The conjugate posterior mean is y * tau^2 / (tau^2 + sigma^2)."""
    np.random.seed(2)
    prior_samples = np.random.normal(0.0, TAU, size=(1000, 2))
    result = mc.TMCMC(n_particles=1000, n_mcmc_steps=5).run(
        conjugate_problem, prior_samples=prior_samples
    )
    expected = Y_OBS * TAU**2 / (TAU**2 + SIGMA_OBS**2)
    assert np.abs(result.mean() - expected).max() < 0.25


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_rhat_is_near_one_for_converged_chains(gaussian_problem):
    np.random.seed(0)
    multi = mc.run_chains(
        mc.MetropolisHastings(proposal_cov=SIGMA * 1.2, n_samples=8000),
        gaussian_problem,
        x0=[0.0, 0.0],
        n_chains=4,
    )
    chains = [c.discard(2000) for c in multi]      # MultiChainResult is iterable
    rhat = mc.gelman_rubin(chains)
    assert np.all(rhat < 1.1), f"converged chains flagged as not converged: {rhat}"

    # the object's own method must agree with the free function
    assert np.allclose(multi.gelman_rubin(), mc.gelman_rubin(list(multi)))


def test_rhat_detects_chains_that_have_not_mixed():
    """Chains stuck in different places must not be reported as converged."""
    rng = np.random.default_rng(0)
    stuck = [
        rng.normal(loc, 0.01, size=(2000, 2))       # four tight, well-separated blobs
        for loc in (-30.0, -10.0, 10.0, 30.0)
    ]
    rhat = mc.gelman_rubin(stuck)
    assert np.all(rhat > 1.5), f"unmixed chains reported as converged: {rhat}"


def test_ess_is_bounded_by_the_sample_count(gaussian_problem):
    np.random.seed(0)
    n = 5000
    result = mc.MetropolisHastings(proposal_cov=SIGMA * 1.2, n_samples=n).run(
        gaussian_problem, x0=[0.0, 0.0]
    ).discard(1000)
    ess = result.ess()
    assert np.all(ess > 0)
    assert np.all(ess <= n - 1000 + 1e-9), f"ESS exceeds the number of samples: {ess}"


def test_correlated_chain_has_lower_ess_than_an_independent_one(gaussian_problem):
    """A tiny proposal makes a highly autocorrelated chain; ESS must reflect that."""
    np.random.seed(0)
    tuned = mc.MetropolisHastings(proposal_cov=SIGMA * 1.2, n_samples=10_000).run(
        gaussian_problem, x0=[0.0, 0.0]
    ).discard(2000)

    np.random.seed(0)
    sticky = mc.MetropolisHastings(proposal_cov=SIGMA * 1e-3, n_samples=10_000).run(
        gaussian_problem, x0=list(MU)
    ).discard(2000)

    assert np.min(sticky.ess()) < np.min(tuned.ess())
