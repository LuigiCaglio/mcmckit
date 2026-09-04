"""Correctness tests for GaussianNoiseLikelihood.

Each of the three noise modes is checked against a closed form, not merely
executed: fixed sigma against the Gaussian log density, estimated sigma
against the analytic maximum-likelihood sigma, and marginalised sigma against
the Inverse-Gamma integral it claims to compute.
"""

import numpy as np
import pytest
from scipy import stats

import mcmckit as mc


# ----------------------------------------------------------------------
# A linear forward model, so every quantity has an analytic answer
# ----------------------------------------------------------------------

X = np.linspace(0.0, 5.0, 12)


def forward(theta):
    """y = a*x + b."""
    return theta[0] * X + theta[1]


TRUE = np.array([2.0, -1.0])
Y_CLEAN = forward(TRUE)

rng = np.random.default_rng(0)
Y_OBS = Y_CLEAN + rng.normal(0.0, 0.3, X.size)


def residuals(theta):
    return Y_OBS - forward(theta)


# ======================================================================
# Fixed sigma
# ======================================================================

def test_fixed_sigma_matches_the_gaussian_log_density():
    """Differs from the exact log pdf only by the -n/2 log(2*pi) constant.

    That constant does not depend on theta, so dropping it is legitimate for
    sampling. The test pins the difference so it cannot silently become
    theta-dependent.
    """
    sigma = 0.3
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=sigma)

    for theta in ([2.0, -1.0], [1.5, 0.4], [2.7, -2.0]):
        exact = stats.norm.logpdf(Y_OBS, forward(np.array(theta)), sigma).sum()
        offset = 0.5 * Y_OBS.size * np.log(2 * np.pi)
        assert np.isclose(ll(theta), exact + offset)


def test_scalar_and_per_observation_sigma_agree():
    a = mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=0.3)
    b = mc.GaussianNoiseLikelihood(forward, Y_OBS,
                                   noise_std=np.full(X.size, 0.3))
    assert np.isclose(a(TRUE), b(TRUE))


def test_per_channel_sigma_applies_to_the_right_group():
    """Two channels with different sigma must equal the hand-built version."""
    groups = [np.arange(0, 6), np.arange(6, 12)]
    sig = [0.2, 0.7]

    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=sig, groups=groups)

    per_obs = np.empty(X.size)
    per_obs[groups[0]] = sig[0]
    per_obs[groups[1]] = sig[1]
    reference = mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=per_obs)

    assert np.isclose(ll(TRUE), reference(TRUE))
    # and it is genuinely different from using one sigma everywhere
    flat = mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=0.2)
    assert not np.isclose(ll(TRUE), flat(TRUE))


def test_larger_residuals_are_less_likely():
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=0.3)
    assert ll(TRUE) > ll([2.0, -1.5])
    assert ll(TRUE) > ll([3.0, -1.0])


# ----- input validation -----------------------------------------------

def test_non_positive_sigma_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=0.0)
    with pytest.raises(ValueError, match="positive"):
        mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=-1.0)


def test_groups_must_partition_the_observations():
    with pytest.raises(ValueError, match="partition"):
        mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=0.3,
                                   groups=[[0, 1, 2]])          # incomplete
    with pytest.raises(ValueError, match="partition"):
        mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=0.3,
                                   groups=[np.arange(12), [0]])  # overlapping


def test_wrong_length_sigma_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=[0.1, 0.2, 0.3])


def test_marginalise_conflicts_with_a_fixed_sigma():
    with pytest.raises(ValueError, match="incompatible"):
        mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=0.3,
                                   marginalise_noise=True)


# ======================================================================
# Estimated sigma
# ======================================================================

def test_estimated_mode_reports_its_extra_parameters():
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS)
    assert ll.estimate_noise is True
    assert ll.marginalise_noise is False
    assert ll.n_noise_params == 1

    groups = [np.arange(0, 6), np.arange(6, 12)]
    ll2 = mc.GaussianNoiseLikelihood(forward, Y_OBS, groups=groups)
    assert ll2.n_noise_params == 2


def test_estimated_at_a_given_sigma_equals_fixed_at_that_sigma():
    sigma = 0.42
    free = mc.GaussianNoiseLikelihood(forward, Y_OBS)
    fixed = mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=sigma)

    assert np.isclose(free([*TRUE, np.log(sigma)]), fixed(TRUE))


def test_estimated_sigma_peaks_at_the_analytic_mle():
    """Maximising over log-sigma must give sqrt(RSS/n)."""
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS)
    r = residuals(TRUE)
    mle = np.sqrt(r @ r / r.size)

    grid = np.log(np.linspace(0.5 * mle, 2.0 * mle, 400))
    values = [ll([*TRUE, ls]) for ls in grid]
    best = np.exp(grid[int(np.argmax(values))])

    assert np.isclose(best, mle, rtol=2e-2)


def test_estimated_noise_is_recovered_by_sampling():
    """End to end: sample slope, intercept and log-sigma together."""
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS)

    def log_post(theta):
        if not (-5 < theta[2] < 2):          # prior on log sigma
            return -np.inf
        return ll(theta)

    res = mc.ram(log_post, [1.0, 0.0, np.log(0.5)], 20_000,
                 initial_cov=0.01, param_names=["a", "b", "log_sigma"],
                 rng=np.random.default_rng(0)).discard(4000)

    est = res.mean()
    assert np.allclose(est[:2], TRUE, atol=0.25)

    r = residuals(TRUE)
    assert np.isclose(np.exp(est[2]), np.sqrt(r @ r / r.size), rtol=0.35)


# ======================================================================
# Marginalised sigma
# ======================================================================

def test_marginalised_matches_the_inverse_gamma_integral():
    """Must equal -(alpha + n/2) * log(beta + RSS/2), summed over channels."""
    alpha, beta = 2.0, 0.05
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS, marginalise_noise=True,
                                    inv_gamma_alpha=alpha, inv_gamma_beta=beta)

    for theta in ([2.0, -1.0], [1.2, 0.5]):
        r = residuals(np.array(theta))
        expected = -(alpha + 0.5 * r.size) * np.log(beta + 0.5 * (r @ r))
        assert np.isclose(ll(theta), expected)


def test_marginalised_matches_numerical_integration_over_sigma():
    """The analytic form must equal integrating sigma out by quadrature.

    Both sides drop the same theta-independent constants, so they are
    compared as a difference between two theta values.
    """
    alpha, beta = 3.0, 0.05
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS, marginalise_noise=True,
                                    inv_gamma_alpha=alpha, inv_gamma_beta=beta)

    def numeric(theta):
        r = residuals(np.array(theta))
        v = np.linspace(1e-6, 4.0, 400_000)          # v = sigma^2
        log_lik = -0.5 * (r @ r) / v - 0.5 * r.size * np.log(v)
        log_prior = stats.invgamma.logpdf(v, a=alpha, scale=beta)
        integrand = np.exp(log_lik + log_prior - np.max(log_lik + log_prior))
        return (np.log(np.trapezoid(integrand, v))
                + np.max(log_lik + log_prior))

    t1, t2 = [2.0, -1.0], [1.4, 0.3]
    assert np.isclose(ll(t1) - ll(t2), numeric(t1) - numeric(t2), atol=1e-3)


def test_posterior_sigma_matches_the_inverse_gamma_posterior_mean():
    alpha, beta = 2.0, 0.05
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS, marginalise_noise=True,
                                    inv_gamma_alpha=alpha, inv_gamma_beta=beta)

    r = residuals(TRUE)
    a = alpha + 0.5 * r.size
    b = beta + 0.5 * (r @ r)
    assert np.allclose(ll.posterior_sigma(TRUE), np.sqrt(b / (a - 1)))


def test_posterior_sigma_is_close_to_the_true_noise():
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS, marginalise_noise=True,
                                    inv_gamma_alpha=2.0, inv_gamma_beta=0.05)
    assert np.isclose(ll.posterior_sigma(TRUE)[0], 0.3, atol=0.15)


def test_posterior_sigma_requires_marginalised_mode():
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=0.3)
    with pytest.raises(RuntimeError, match="marginalised"):
        ll.posterior_sigma(TRUE)


def test_marginalised_per_channel_uses_separate_priors():
    groups = [np.arange(0, 6), np.arange(6, 12)]
    ll = mc.GaussianNoiseLikelihood(
        forward, Y_OBS, groups=groups, marginalise_noise=True,
        inv_gamma_alpha=[2.0, 5.0], inv_gamma_beta=[0.05, 0.5],
    )
    r = residuals(TRUE)
    expected = 0.0
    for k, (a0, b0) in enumerate([(2.0, 0.05), (5.0, 0.5)]):
        rk = r[groups[k]]
        expected -= (a0 + 0.5 * rk.size) * np.log(b0 + 0.5 * (rk @ rk))
    assert np.isclose(ll(TRUE), expected)

    assert ll.posterior_sigma(TRUE).shape == (2,)


def test_marginalised_recovers_the_model_parameters():
    ll = mc.GaussianNoiseLikelihood(forward, Y_OBS, marginalise_noise=True,
                                    inv_gamma_alpha=2.0, inv_gamma_beta=0.05)
    res = mc.ram(ll, [0.0, 0.0], 15_000, initial_cov=0.05,
                 rng=np.random.default_rng(1)).discard(3000)
    assert np.allclose(res.mean(), TRUE, atol=0.25)


def test_the_three_modes_agree_on_where_the_optimum_is():
    """Different constants, same location of the best-fitting theta."""
    fixed = mc.GaussianNoiseLikelihood(forward, Y_OBS, noise_std=0.3)
    marg = mc.GaussianNoiseLikelihood(forward, Y_OBS, marginalise_noise=True,
                                      inv_gamma_alpha=2.0, inv_gamma_beta=0.05)

    grid = np.linspace(1.5, 2.5, 200)
    best_fixed = grid[int(np.argmax([fixed([a, -1.0]) for a in grid]))]
    best_marg = grid[int(np.argmax([marg([a, -1.0]) for a in grid]))]
    assert np.isclose(best_fixed, best_marg, atol=0.02)
