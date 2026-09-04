"""Tests for ModelComparison weights and BMAResult.

`bayes_factor` itself is covered in test_model_comparison.py. This file covers
the two classes built on top of it, which turn log-evidences into posterior
model probabilities and a mixture predictive.

The weight arithmetic is tested against injected log-evidences rather than
real TMCMC runs, so the maths is checked exactly and quickly; one end-to-end
run at the bottom confirms the pieces fit together.
"""

import numpy as np
import pytest

import mcmckit as mc
from mcmckit.core.model_comparison import BMAResult


def comparison_with(log_evidences, names=None):
    """A ModelComparison with results injected, bypassing TMCMC."""
    names = names or [f"M{i}" for i in range(len(log_evidences))]
    cmp = mc.ModelComparison(models=[])
    cmp._results = [
        {"name": n, "log_evidence": float(le), "result": None, "sampler": None}
        for n, le in zip(names, log_evidences)
    ]
    cmp._results.sort(key=lambda r: r["log_evidence"], reverse=True)

    # run() also derives each model's Bayes factor against the best one;
    # summary() reads those keys, so mirror it here.
    best = cmp._results[0]["log_evidence"]
    for r in cmp._results:
        info = mc.bayes_factor(r["log_evidence"], best)
        r["log_bf_vs_best"] = info["log_bf"]
        r["log10_bf_vs_best"] = info["log10_bf"]
        r["evidence_vs_best"] = info["evidence"]
    return cmp


# ======================================================================
# ModelComparison.weights
# ======================================================================

def test_weights_sum_to_one():
    w = comparison_with([-10.0, -12.0, -20.0]).weights()
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w > 0)


def test_equal_evidence_gives_uniform_weights():
    w = comparison_with([-7.0, -7.0, -7.0, -7.0]).weights()
    assert np.allclose(w, 0.25)


def test_weight_ratio_is_the_evidence_ratio():
    """w1/w2 must equal exp(logE1 - logE2)."""
    w = comparison_with([-10.0, -12.5]).weights()
    assert np.isclose(w[0] / w[1], np.exp(-10.0 - (-12.5)))


def test_weights_are_ordered_best_first():
    cmp = comparison_with([-30.0, -10.0, -20.0], names=["c", "a", "b"])
    w = cmp.weights()
    assert [r["name"] for r in cmp._results] == ["a", "b", "c"]
    assert w[0] > w[1] > w[2]
    assert cmp.best_model() == "a"


def test_weights_survive_enormous_evidence_gaps():
    """A 700-nat gap overflows a naive exp; the shift must prevent it."""
    w = comparison_with([-10.0, -1000.0]).weights()
    assert np.isfinite(w).all()
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w[0], 1.0)


def test_prior_weights_shift_the_posterior():
    """Halving one model's prior must halve its posterior odds."""
    evs = [-10.0, -10.0]
    flat = comparison_with(evs).weights()
    assert np.allclose(flat, 0.5)

    skewed = comparison_with(evs).weights(prior_weights=[2.0, 1.0])
    assert np.isclose(skewed[0] / skewed[1], 2.0)
    assert np.isclose(skewed.sum(), 1.0)


def test_prior_weights_are_normalised_internally():
    a = comparison_with([-10.0, -11.0]).weights(prior_weights=[2.0, 1.0])
    b = comparison_with([-10.0, -11.0]).weights(prior_weights=[200.0, 100.0])
    assert np.allclose(a, b)


def test_prior_weights_length_is_validated():
    with pytest.raises(ValueError, match="prior_weights"):
        comparison_with([-10.0, -11.0]).weights(prior_weights=[1.0, 1.0, 1.0])


def test_weights_before_run_is_an_error():
    with pytest.raises(RuntimeError, match="run"):
        mc.ModelComparison(models=[]).weights()


def test_summary_reports_each_model_against_the_best():
    cmp = comparison_with([-10.0, -14.0], names=["good", "bad"])
    rows = cmp.summary(print_table=False)
    assert [r["name"] for r in rows] == ["good", "bad"]
    assert np.isclose(rows[0]["log_bf_vs_best"], 0.0)
    assert np.isclose(rows[1]["log_bf_vs_best"], -4.0)


# ======================================================================
# BMAResult
# ======================================================================

def bma_fixture():
    """Two models: one predicting ~0, one predicting ~10, weights 0.75/0.25."""
    rng = np.random.default_rng(0)
    p1 = rng.normal(0.0, 1.0, size=(300, 4))
    p2 = rng.normal(10.0, 1.0, size=(100, 4))
    return BMAResult([p1, p2], np.array([0.75, 0.25]), ["low", "high"])


def test_bma_predictions_are_the_pooled_mixture():
    bma = bma_fixture()
    assert bma.predictions.shape == (400, 4)
    assert np.allclose(bma.predictions[:300], bma.model_predictions[0])
    assert np.allclose(bma.predictions[300:], bma.model_predictions[1])


def test_bma_statistics_match_the_pooled_sample():
    bma = bma_fixture()
    assert np.allclose(bma.mean(), bma.predictions.mean(axis=0))
    assert np.allclose(bma.std(), bma.predictions.std(axis=0))
    assert np.allclose(bma.quantile(0.5), np.median(bma.predictions, axis=0))


def test_bma_mean_sits_between_the_component_means():
    """A 75/25 mixture of 0 and 10 must land near 2.5, not at either mode."""
    bma = bma_fixture()
    assert np.all(bma.mean() > 1.5)
    assert np.all(bma.mean() < 3.5)


def test_bma_quantiles_are_ordered():
    bma = bma_fixture()
    lo, hi = bma.quantile([0.05, 0.95])
    assert np.all(lo < hi)


def test_bma_decompose_reports_each_model_separately():
    bma = bma_fixture()
    parts = bma.decompose()

    assert set(parts) == {"low", "high"}
    assert np.isclose(parts["low"]["weight"], 0.75)
    assert np.isclose(parts["high"]["weight"], 0.25)
    assert np.allclose(parts["low"]["mean"], bma.model_predictions[0].mean(axis=0))
    assert np.allclose(parts["high"]["mean"], bma.model_predictions[1].mean(axis=0))
    # the components are genuinely distinct
    assert np.all(parts["high"]["mean"] - parts["low"]["mean"] > 5.0)


def test_bma_is_wider_than_either_component():
    """Averaging over models adds between-model spread; that is the point."""
    bma = bma_fixture()
    parts = bma.decompose()
    assert np.all(bma.std() > parts["low"]["std"])
    assert np.all(bma.std() > parts["high"]["std"])


# ======================================================================
# End to end
# ======================================================================

def test_model_comparison_prefers_the_model_that_generated_the_data():
    """A quadratic truth: the quadratic model must beat the linear one."""
    x = np.linspace(-2.0, 2.0, 25)
    rng = np.random.default_rng(4)
    y = 1.0 + 0.5 * x + 2.0 * x**2 + rng.normal(0.0, 0.2, x.size)
    sigma = 0.2

    def make(n_params, design):
        def ll(theta):
            r = y - design(theta)
            return -0.5 * np.sum((r / sigma) ** 2)
        problem = mc.Problem(prior=lambda t: 0.0, likelihood=ll)
        prior = np.random.default_rng(0).uniform(-4, 4, size=(400, n_params))
        return problem, prior

    linear_p, linear_s = make(2, lambda t: t[0] + t[1] * x)
    quad_p, quad_s = make(3, lambda t: t[0] + t[1] * x + t[2] * x**2)

    cmp = mc.ModelComparison(
        models=[("linear", linear_p, linear_s), ("quadratic", quad_p, quad_s)],
        tmcmc_kwargs={"n_particles": 400, "n_mcmc_steps": 2},
    ).run()

    assert cmp.best_model() == "quadratic"

    w = cmp.weights()
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > 0.99          # the truth should win overwhelmingly
