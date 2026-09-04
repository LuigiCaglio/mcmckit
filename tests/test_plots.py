"""Plotting checks.

These verify that every drawing method runs and produces a figure with the
expected number of axes. They cannot judge whether a plot looks right, but
they do catch the failure that actually happens: a plotting call that raises
because an argument, a style name or an array shape changed underneath it.

Matplotlib is forced onto a non-interactive backend so nothing opens a window.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mcmckit as mc
from mcmckit.core.model_comparison import BMAResult


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def result():
    mu = np.array([1.0, -2.0])
    prec = np.linalg.inv([[1.0, 0.6], [0.6, 1.0]])

    def log_post(theta):
        d = theta - mu
        return -0.5 * d @ prec @ d

    return mc.ram(log_post, [0.0, 0.0], 3000, initial_cov=0.2,
                  param_names=["x", "y"],
                  rng=np.random.default_rng(0)).discard(500)


# ======================================================================
# Result
# ======================================================================

def test_plot_trace_draws_one_panel_per_parameter(result):
    result.plot_trace(title="trace")
    assert len(plt.gcf().axes) == 2


def test_plot_marginals_draws_one_panel_per_parameter(result):
    result.plot_marginals(bins=20)
    assert len(plt.gcf().axes) >= 2


def test_plot_autocorr_runs(result):
    result.plot_autocorr(max_lag=30)
    assert plt.gcf().axes


@pytest.mark.parametrize("style", ["corner", "scatter", "full", "kde"])
def test_every_corner_style_draws_a_full_grid(style, result):
    fig = result.plot_corner(style=style, title=style)
    assert fig is not None
    # a 2-parameter corner is a 2x2 grid, whatever the style fills in
    assert len(fig.axes) == 4


def test_corner_accepts_true_values(result):
    fig = result.plot_corner(true_values=[1.0, -2.0])
    assert fig is not None


def test_corner_forwards_style_keywords(result):
    fig = result.plot_corner(style="scatter",
                             scatter_kwargs={"s": 3, "color": "k"},
                             hist_kwargs={"color": "grey"})
    assert fig is not None


def test_plotting_works_without_parameter_names():
    """A chain built by hand has no names; plots must still label the axes."""
    plain = mc.Result(samples=np.random.default_rng(0).normal(size=(400, 2)))
    plain.plot_trace()
    assert len(plt.gcf().axes) == 2
    assert plain.plot_corner() is not None


def test_single_parameter_result_plots(result):
    one = result.select([0])
    one.plot_trace()
    assert len(plt.gcf().axes) == 1
    assert one.plot_corner() is not None


# ======================================================================
# Posterior predictive
# ======================================================================

def test_posterior_predictive_bands(result):
    x = np.linspace(0, 1, 15)

    def forward(theta):
        return theta[0] * x + theta[1]

    pred = result.posterior_predictive(forward, n_eval=100)
    assert pred.predictions.shape == (100, 15)
    assert np.all(pred.quantile(0.05) < pred.quantile(0.95))

    pred.plot_bands(x=x, y_obs=forward([1.0, -2.0]), xlabel="x", ylabel="y")
    assert plt.gcf().axes


def test_posterior_predictive_evaluates_every_sample_by_default(result):
    def forward(theta):
        return np.array([theta[0], theta[1], theta[0] + theta[1]])

    pred = result.posterior_predictive(forward)
    assert pred.predictions.shape == (len(result.samples), 3)
    # the predictive of a linear map must match the map of the mean
    assert np.allclose(pred.mean(), forward(result.mean()), atol=1e-8)


# ======================================================================
# BMA
# ======================================================================

def test_bma_plots():
    rng = np.random.default_rng(0)
    bma = BMAResult([rng.normal(0, 1, (200, 6)), rng.normal(4, 1, (100, 6))],
                    np.array([0.7, 0.3]), ["A", "B"])

    bma.plot_bands(x=np.arange(6), title="bma")
    assert plt.gcf().axes

    bma.plot_decompose(x=np.arange(6), title="decompose")
    assert plt.gcf().axes


# ======================================================================
# TMCMC
# ======================================================================

def test_tmcmc_plot_stages():
    def log_lik(theta):
        return -0.5 * np.sum((theta - 1.0) ** 2)

    problem = mc.Problem(prior=lambda t: 0.0, likelihood=log_lik,
                         param_names=["a", "b"])
    prior_samples = np.random.default_rng(0).uniform(-5, 5, size=(200, 2))

    sampler = mc.TMCMC(n_particles=200, n_mcmc_steps=2)
    sampler.run(problem, prior_samples=prior_samples)

    fig = sampler.plot_stages(max_stages=4)
    assert fig is not None
    assert fig.axes


# ======================================================================
# Multi-chain
# ======================================================================

def test_multichain_plot_traces():
    def log_post(theta):
        return -0.5 * np.sum(theta**2)

    multi = mc.run_chains(mc.RAM(n_samples=800, initial_cov=0.5),
                          mc.Problem(prior=lambda t: 0.0, likelihood=log_post,
                                     param_names=["p", "q"]),
                          x0=[[0.0, 0.0], [1.0, 1.0]], n_chains=2)
    multi.plot_traces(title="chains")
    assert plt.gcf().axes
