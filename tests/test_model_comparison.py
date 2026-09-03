"""Bayes factors: the arithmetic, the labels, and one analytic check."""

import numpy as np
import pytest

import mcmckit as mc


def test_bayes_factor_is_the_evidence_ratio():
    out = mc.bayes_factor(-10.0, -12.0)
    assert np.isclose(out["log_bf"], 2.0)
    assert np.isclose(out["bf"], np.exp(2.0))
    assert np.isclose(out["log10_bf"], 2.0 / np.log(10))


def test_equal_evidence_prefers_neither_strongly():
    out = mc.bayes_factor(-5.0, -5.0)
    assert np.isclose(out["log_bf"], 0.0)
    assert np.isclose(out["bf"], 1.0)
    assert out["evidence"] == "Barely worth mentioning"


def test_swapping_the_models_negates_the_log_bayes_factor():
    a, b = mc.bayes_factor(-3.0, -9.0), mc.bayes_factor(-9.0, -3.0)
    assert np.isclose(a["log_bf"], -b["log_bf"])
    assert np.isclose(a["bf"], 1.0 / b["bf"])
    assert a["preferred"] == "M1" and b["preferred"] == "M2"
    assert a["evidence"] == b["evidence"], "strength of evidence is symmetric"


@pytest.mark.parametrize(
    "log10_bf,expected",
    [
        (0.1, "Barely worth mentioning"),
        (0.7, "Substantial"),
        (1.5, "Strong"),
        (2.5, "Decisive"),
        (0.5, "Substantial"),      # on the boundary: >= is inclusive
        (1.0, "Strong"),
        (2.0, "Decisive"),
    ],
)
def test_jeffreys_labels(log10_bf, expected):
    log_ev_1 = log10_bf * np.log(10)
    assert mc.bayes_factor(log_ev_1, 0.0)["evidence"] == expected


def test_labels_use_the_magnitude_not_the_sign():
    """Evidence 100:1 against M1 is just as decisive as 100:1 for it."""
    assert mc.bayes_factor(0.0, 2.5 * np.log(10))["evidence"] == "Decisive"


def test_enormous_evidence_ratios_do_not_overflow():
    out = mc.bayes_factor(1e5, -1e5)
    assert np.isfinite(out["bf"]), "bf must be capped rather than inf"
    assert np.isfinite(out["log_bf"])
    assert out["preferred"] == "M1"


def test_bayes_factor_of_two_conjugate_models_matches_the_analytic_value():
    """A real check, not just arithmetic on two numbers I made up.

    Two nested Gaussian models for the same datum, both with closed-form
    evidence: p(y) = N(y | 0, tau^2 + sigma^2). The Bayes factor between them
    is therefore known exactly, and TMCMC's estimate must reproduce it.
    """
    y_obs = np.array([1.4])
    sigma = 1.0
    tau_1, tau_2 = 3.0, 0.5          # a broad and a tight prior

    def analytic_log_evidence(tau):
        v = tau**2 + sigma**2
        return float(np.sum(-0.5 * np.log(2 * np.pi * v) - y_obs**2 / (2 * v)))

    def make_problem(tau):
        def log_prior(theta):
            theta = np.asarray(theta, dtype=float)
            return float(np.sum(-0.5 * np.log(2 * np.pi * tau**2) - theta**2 / (2 * tau**2)))

        def log_likelihood(theta):
            theta = np.asarray(theta, dtype=float)
            return float(
                np.sum(-0.5 * np.log(2 * np.pi * sigma**2)
                       - (y_obs - theta) ** 2 / (2 * sigma**2))
            )

        return mc.Problem(prior=log_prior, likelihood=log_likelihood, param_names=["x"])

    estimates = []
    for tau in (tau_1, tau_2):
        np.random.seed(0)
        prior_samples = np.random.normal(0.0, tau, size=(1500, 1))
        result = mc.TMCMC(n_particles=1500, n_mcmc_steps=5).run(
            make_problem(tau), prior_samples=prior_samples
        )
        estimates.append(result.log_evidence)

    expected_log_bf = analytic_log_evidence(tau_1) - analytic_log_evidence(tau_2)
    got = mc.bayes_factor(estimates[0], estimates[1])

    assert abs(got["log_bf"] - expected_log_bf) < 0.3, (
        f"log BF {got['log_bf']:.3f} vs analytic {expected_log_bf:.3f}"
    )
