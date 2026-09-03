"""HierarchicalProblem: parameter layout, the joint density, and shrinkage.

The mechanical part - splitting and reassembling the parameter vector - is easy
to get subtly wrong and silently mislabels every result if it is. The
statistical part is checked against a conjugate normal-normal model, where the
population posterior and the shrinkage of each group are known in closed form.
"""

import numpy as np
import pytest

import mcmckit as mc

# ---------------------------------------------------------------------------
# Conjugate normal-normal model
#
#   mu ~ N(0, tau_mu^2)                      population mean (the hyperparameter)
#   theta_j | mu ~ N(mu, tau^2)              group means
#   y_j | theta_j ~ N(theta_j, sigma^2 / n)  group sample mean
#
# Marginally y_j | mu ~ N(mu, tau^2 + s_j^2), which makes the posterior of mu
# analytic, and each theta_j shrinks toward mu by a known factor.
# ---------------------------------------------------------------------------

TAU_MU = 10.0          # broad hyperprior on the population mean
TAU = 1.0              # between-group spread
S_OBS = 0.5            # within-group standard error
Y_OBS = np.array([2.0, 3.0, 4.0, 5.0])
N_GROUPS = len(Y_OBS)


def hyperprior(phi):
    return float(-0.5 * np.log(2 * np.pi * TAU_MU**2) - phi[0] ** 2 / (2 * TAU_MU**2))


def group_prior(theta_j, phi):
    return float(-0.5 * np.log(2 * np.pi * TAU**2) - (theta_j[0] - phi[0]) ** 2 / (2 * TAU**2))


def make_group_likelihood(y):
    def log_likelihood(theta_j):
        return float(-0.5 * np.log(2 * np.pi * S_OBS**2)
                     - (y - theta_j[0]) ** 2 / (2 * S_OBS**2))
    return log_likelihood


@pytest.fixture
def problem():
    return mc.HierarchicalProblem(
        hyperprior=hyperprior,
        group_prior=group_prior,
        group_likelihoods=[make_group_likelihood(y) for y in Y_OBS],
        n_hyper=1,
        n_group=1,
        param_names_hyper=["mu"],
        param_names_group=["theta"],
    )


def analytic_mu_posterior():
    """mu | y with all groups marginalised: Gaussian, since everything is."""
    v = TAU**2 + S_OBS**2
    precision = 1.0 / TAU_MU**2 + N_GROUPS / v
    mean = (Y_OBS.sum() / v) / precision
    return mean, 1.0 / precision


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------

def test_n_params_and_names(problem):
    assert problem.n_params == 1 + N_GROUPS * 1
    assert problem.n_groups == N_GROUPS
    assert problem.param_names[0] == "mu"
    assert problem.param_names[1:] == [f"theta_{j}" for j in range(N_GROUPS)]


def test_split_recovers_the_pieces(problem):
    theta = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    phi, groups = problem.split(theta)
    assert np.allclose(phi, [0.5])
    assert len(groups) == N_GROUPS
    for j, g in enumerate(groups):
        assert np.allclose(g, [theta[1 + j]])


def test_default_x0_round_trips_through_split(problem):
    """Assembling then splitting must be the identity, or every result is mislabelled."""
    phi0 = [0.7]
    group_x0s = [[1.1], [2.2], [3.3], [4.4]]
    x0 = problem.default_x0(phi0, group_x0s)
    assert x0.shape == (problem.n_params,)
    phi, groups = problem.split(x0)
    assert np.allclose(phi, phi0)
    for got, want in zip(groups, group_x0s):
        assert np.allclose(got, want)


def test_split_handles_multidimensional_groups():
    prob = mc.HierarchicalProblem(
        hyperprior=lambda phi: 0.0,
        group_prior=lambda t, phi: 0.0,
        group_likelihoods=[lambda t: 0.0, lambda t: 0.0],
        n_hyper=2,
        n_group=3,
    )
    assert prob.n_params == 2 + 2 * 3
    theta = np.arange(8, dtype=float)
    phi, groups = prob.split(theta)
    assert np.allclose(phi, [0, 1])
    assert np.allclose(groups[0], [2, 3, 4])
    assert np.allclose(groups[1], [5, 6, 7])


def test_a_single_callable_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="list of callables"):
        mc.HierarchicalProblem(
            hyperprior=lambda phi: 0.0,
            group_prior=lambda t, phi: 0.0,
            group_likelihoods=lambda t: 0.0,     # not a list
            n_hyper=1,
            n_group=1,
        )


# ---------------------------------------------------------------------------
# The joint density
# ---------------------------------------------------------------------------

def test_log_posterior_is_prior_plus_likelihood(problem):
    theta = np.array([3.0, 2.1, 3.1, 3.9, 4.8])
    assert np.isclose(
        problem.log_posterior(theta),
        problem.log_prior(theta) + problem.log_likelihood(theta),
    )


def test_log_prior_sums_the_hyperprior_and_every_group(problem):
    theta = np.array([3.0, 2.1, 3.1, 3.9, 4.8])
    phi, groups = problem.split(theta)
    expected = hyperprior(phi) + sum(group_prior(g, phi) for g in groups)
    assert np.isclose(problem.log_prior(theta), expected)


def test_log_likelihood_sums_every_group(problem):
    theta = np.array([3.0, 2.1, 3.1, 3.9, 4.8])
    _, groups = problem.split(theta)
    expected = sum(make_group_likelihood(y)(g) for y, g in zip(Y_OBS, groups))
    assert np.isclose(problem.log_likelihood(theta), expected)


def test_an_impossible_hyperparameter_makes_the_posterior_minus_infinity():
    prob = mc.HierarchicalProblem(
        hyperprior=lambda phi: -np.inf if phi[0] < 0 else 0.0,
        group_prior=lambda t, phi: 0.0,
        group_likelihoods=[lambda t: 0.0],
        n_hyper=1,
        n_group=1,
    )
    assert prob.log_posterior([-1.0, 0.0]) == -np.inf
    assert np.isfinite(prob.log_posterior([1.0, 0.0]))


# ---------------------------------------------------------------------------
# Statistics: does sampling it give the right answer?
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sampled():
    prob = mc.HierarchicalProblem(
        hyperprior=hyperprior,
        group_prior=group_prior,
        group_likelihoods=[make_group_likelihood(y) for y in Y_OBS],
        n_hyper=1,
        n_group=1,
        param_names_hyper=["mu"],
        param_names_group=["theta"],
    )
    np.random.seed(0)
    x0 = prob.default_x0([Y_OBS.mean()], [[y] for y in Y_OBS])
    problem_obj = mc.Problem(
        prior=prob.log_prior, likelihood=prob.log_likelihood, param_names=prob.param_names
    )
    result = mc.DRAM(n_samples=20_000).run(problem_obj, x0=x0).discard(4_000)
    return prob, result


def test_population_mean_matches_the_analytic_posterior(sampled):
    prob, result = sampled
    mean_expected, var_expected = analytic_mu_posterior()
    mu_samples = result.samples[:, 0]
    assert abs(mu_samples.mean() - mean_expected) < 0.1, (
        f"mu posterior mean {mu_samples.mean():.3f} vs analytic {mean_expected:.3f}"
    )
    assert abs(mu_samples.std() - np.sqrt(var_expected)) < 0.1


def test_group_estimates_shrink_toward_the_population_mean(sampled):
    """The defining behaviour of a hierarchical model.

    Each group's posterior mean must sit between its own observation and the
    population mean, closer to the population mean than the raw data is.
    """
    prob, result = sampled
    mu_hat = result.samples[:, 0].mean()
    for j, y in enumerate(Y_OBS):
        theta_hat = result.samples[:, 1 + j].mean()
        assert min(y, mu_hat) - 0.05 <= theta_hat <= max(y, mu_hat) + 0.05, (
            f"group {j}: estimate {theta_hat:.3f} not between y={y} and mu={mu_hat:.3f}"
        )
        assert abs(theta_hat - mu_hat) < abs(y - mu_hat) + 1e-9, (
            f"group {j} was not shrunk toward the population mean"
        )


def test_shrinkage_factor_matches_the_analytic_value(sampled):
    """theta_j shrinks toward mu by tau^2 / (tau^2 + s^2), a known factor."""
    prob, result = sampled
    mu_hat = result.samples[:, 0].mean()
    expected_weight = TAU**2 / (TAU**2 + S_OBS**2)      # weight on the observation
    for j, y in enumerate(Y_OBS):
        theta_hat = result.samples[:, 1 + j].mean()
        realised = (theta_hat - mu_hat) / (y - mu_hat)
        assert abs(realised - expected_weight) < 0.1, (
            f"group {j}: shrinkage {realised:.3f} vs analytic {expected_weight:.3f}"
        )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_extract_hyper_takes_the_leading_columns(sampled):
    prob, result = sampled
    hyper = prob.extract_hyper(result)
    assert hyper.samples.shape == (result.samples.shape[0], 1)
    assert np.allclose(hyper.samples[:, 0], result.samples[:, 0])
    assert hyper.param_names == ["mu"]


def test_extract_group_takes_that_group_s_columns(sampled):
    prob, result = sampled
    for j in range(N_GROUPS):
        group = prob.extract_group(result, j)
        assert group.samples.shape == (result.samples.shape[0], 1)
        assert np.allclose(group.samples[:, 0], result.samples[:, 1 + j])
        assert group.param_names == [f"theta_{j}"]


def test_extract_group_rejects_an_out_of_range_index(sampled):
    prob, result = sampled
    for bad in (-1, N_GROUPS, N_GROUPS + 5):
        with pytest.raises(ValueError, match="must be in"):
            prob.extract_group(result, bad)
