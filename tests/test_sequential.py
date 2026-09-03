"""Sequential updating: turning a posterior into the next step's prior.

The claim that matters is that splitting the data changes nothing. Updating on
batch 1 and then on batch 2 must land in the same place as updating on both at
once, because

    p(theta | y1, y2)  proportional to  p(y2 | theta) p(theta | y1)

If `PosteriorPrior` is not a properly normalised density, or fits the wrong
moments, that identity breaks.
"""

import numpy as np
import pytest

import mcmckit as mc
from mcmckit.core.sequential import PosteriorPrior


@pytest.fixture
def gaussian_samples():
    """20k draws from a known correlated 2-D Gaussian."""
    mean = np.array([1.0, -2.0])
    cov = np.array([[2.0, 0.8], [0.8, 1.0]])
    rng = np.random.default_rng(0)
    return rng.multivariate_normal(mean, cov, size=20_000), mean, cov


# ---------------------------------------------------------------------------
# Density fitting
# ---------------------------------------------------------------------------

def test_gaussian_fit_recovers_the_moments(gaussian_samples):
    samples, mean, cov = gaussian_samples
    prior = PosteriorPrior(samples, method="gaussian")
    assert np.allclose(prior.mean, mean, atol=0.05)
    assert np.allclose(prior.cov, cov, atol=0.05)
    assert prior.n_params == 2
    assert prior.method == "gaussian"


def test_gaussian_log_density_matches_scipy(gaussian_samples):
    """It must be a *normalised* log-density, not just the quadratic form."""
    scipy_stats = pytest.importorskip("scipy.stats")
    samples, _, _ = gaussian_samples
    prior = PosteriorPrior(samples, method="gaussian")
    reference = scipy_stats.multivariate_normal(mean=prior.mean, cov=prior.cov)
    for theta in ([1.0, -2.0], [0.0, 0.0], [3.0, -4.0], [-2.0, 1.5]):
        assert np.isclose(prior(theta), reference.logpdf(theta), rtol=1e-8)


def test_gaussian_density_integrates_to_one():
    """Normalisation, checked directly on a 1-D grid."""
    rng = np.random.default_rng(1)
    samples = rng.normal(2.0, 0.5, size=(20_000, 1))
    prior = PosteriorPrior(samples, method="gaussian")
    grid = np.linspace(-3.0, 7.0, 4001)
    density = np.exp([prior([x]) for x in grid])
    assert np.isclose(np.trapezoid(density, grid), 1.0, atol=1e-3)


def test_density_is_highest_at_the_mean(gaussian_samples):
    samples, _, _ = gaussian_samples
    prior = PosteriorPrior(samples, method="gaussian")
    at_mean = prior(prior.mean)
    rng = np.random.default_rng(2)
    for _ in range(20):
        offset = prior.mean + rng.standard_normal(2)
        assert prior(offset) <= at_mean


def test_sample_reproduces_the_fitted_distribution(gaussian_samples):
    samples, mean, cov = gaussian_samples
    prior = PosteriorPrior(samples, method="gaussian")
    drawn = prior.sample(50_000, rng=np.random.default_rng(3))
    assert drawn.shape == (50_000, 2)
    assert np.allclose(drawn.mean(axis=0), mean, atol=0.05)
    assert np.allclose(np.cov(drawn.T), cov, atol=0.08)


def test_kde_is_a_valid_density():
    pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(4)
    samples = rng.normal(0.0, 1.0, size=(4000, 1))
    prior = PosteriorPrior(samples, method="kde")
    assert prior.method == "kde"
    grid = np.linspace(-6.0, 6.0, 2001)
    density = np.exp([prior([x]) for x in grid])
    assert np.isclose(np.trapezoid(density, grid), 1.0, atol=1e-2)
    assert prior([0.0]) > prior([3.0]), "density must fall away from the mode"


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="gaussian.*kde"):
        PosteriorPrior(np.zeros((10, 2)), method="histogram")


def test_one_dimensional_samples_are_promoted_to_a_column():
    prior = PosteriorPrior(np.random.default_rng(5).normal(size=500), method="gaussian")
    assert prior.n_params == 1
    assert prior.cov.shape == (1, 1)


# ---------------------------------------------------------------------------
# The property that matters
# ---------------------------------------------------------------------------

def test_splitting_the_data_gives_the_same_posterior():
    """Two sequential updates must match one combined update.

    Conjugate setup, so the combined answer is known in closed form:
    theta ~ N(0, tau^2), y_i | theta ~ N(theta, sigma^2). After n observations
    the posterior is Gaussian with precision 1/tau^2 + n/sigma^2.
    """
    tau, sigma = 3.0, 1.0
    rng = np.random.default_rng(6)
    theta_true = 1.5
    batch_1 = rng.normal(theta_true, sigma, size=12)
    batch_2 = rng.normal(theta_true, sigma, size=8)
    both = np.concatenate([batch_1, batch_2])

    def analytic(data):
        precision = 1.0 / tau**2 + len(data) / sigma**2
        mean = (data.sum() / sigma**2) / precision
        return mean, 1.0 / precision

    def log_prior_flat_gaussian(theta):
        theta = np.asarray(theta, dtype=float)
        return float(-0.5 * np.log(2 * np.pi * tau**2) - theta[0] ** 2 / (2 * tau**2))

    def make_likelihood(data):
        def log_likelihood(theta):
            theta = np.asarray(theta, dtype=float)
            return float(np.sum(-0.5 * np.log(2 * np.pi * sigma**2)
                                - (data - theta[0]) ** 2 / (2 * sigma**2)))
        return log_likelihood

    # --- step 1: batch 1 only
    np.random.seed(0)
    problem_1 = mc.Problem(prior=log_prior_flat_gaussian,
                           likelihood=make_likelihood(batch_1),
                           param_names=["theta"])
    result_1 = mc.DRAM(n_samples=12_000).run(problem_1, x0=[0.0]).discard(2_000)

    mean_1, var_1 = analytic(batch_1)
    assert abs(result_1.mean()[0] - mean_1) < 0.1

    # --- step 2: batch 2, using step 1's posterior as the prior
    prior_2 = result_1.as_prior(method="gaussian")
    np.random.seed(1)
    problem_2 = mc.Problem(prior=prior_2,
                           likelihood=make_likelihood(batch_2),
                           param_names=["theta"])
    result_2 = mc.DRAM(n_samples=12_000, initial_cov=prior_2.cov).run(
        problem_2, x0=prior_2.mean
    ).discard(2_000)

    # --- the combined analytic answer
    mean_both, var_both = analytic(both)

    assert abs(result_2.mean()[0] - mean_both) < 0.1, (
        f"sequential posterior mean {result_2.mean()[0]:.4f} "
        f"vs combined analytic {mean_both:.4f}"
    )
    assert abs(np.sqrt(result_2.cov()[0, 0]) - np.sqrt(var_both)) < 0.05, (
        f"sequential posterior sd {np.sqrt(result_2.cov()[0, 0]):.4f} "
        f"vs combined analytic {np.sqrt(var_both):.4f}"
    )


def test_each_update_tightens_the_posterior():
    """More data must not increase the variance."""
    sigma, tau = 1.0, 5.0
    rng = np.random.default_rng(7)
    data = rng.normal(0.5, sigma, size=30)

    def log_prior(theta):
        theta = np.asarray(theta, dtype=float)
        return float(-0.5 * np.log(2 * np.pi * tau**2) - theta[0] ** 2 / (2 * tau**2))

    prior = log_prior
    spreads = []
    for start in (0, 10, 20):
        chunk = data[start:start + 10]

        def log_likelihood(theta, chunk=chunk):
            theta = np.asarray(theta, dtype=float)
            return float(np.sum(-(chunk - theta[0]) ** 2 / (2 * sigma**2)))

        np.random.seed(start)
        x0 = [0.0] if start == 0 else list(prior.mean)
        problem = mc.Problem(prior=prior, likelihood=log_likelihood, param_names=["t"])
        result = mc.DRAM(n_samples=8_000).run(problem, x0=x0).discard(1_500)
        spreads.append(float(np.sqrt(result.cov()[0, 0])))
        prior = result.as_prior(method="gaussian")

    assert spreads[0] > spreads[1] > spreads[2], f"posterior did not tighten: {spreads}"
