"""Regenerate every figure used in the documentation.

Run:  python tools/make_docs_figures.py

Writes PNGs into docs/images/. Deterministic: every sampler is seeded, so
re-running reproduces byte-comparable figures and the docs stay in step with
the code.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mcmckit as mc

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# --- house style -------------------------------------------------------
# Categorical slots, validated for the light chart surface. Assigned in a
# fixed order, one hue per sampler, never cycled.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd5"
SERIES = {
    "MH":            "#2a78d6",
    "MALA":          "#eb6834",
    "RAM":           "#1baf7a",
    "DRAM":          "#eda100",
    "AdaptiveMALA":  "#e87ba4",
    "Gibbs":         "#008300",
}
TRUTH = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "figure.dpi": 120,
})


def recessive(ax):
    """Push the frame back so the data reads first."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(name, fig=None):
    fig = fig or plt.gcf()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name}")


# ======================================================================
# Shared targets
# ======================================================================

MU = np.array([2.0, -1.0])
COV = np.array([[1.0, 0.8], [0.8, 1.0]])
PREC = np.linalg.inv(COV)


def log_post(theta):
    d = theta - MU
    return -0.5 * d @ PREC @ d


def log_post_and_grad(theta):
    d = theta - MU
    return -0.5 * d @ PREC @ d, -PREC @ d


def banana(theta):
    """Rosenbrock-style curved ridge - a classic hard target."""
    x, y = theta
    return -((1.0 - x) ** 2 + 10.0 * (y - x**2) ** 2) / 2.0


# ======================================================================
# 1. Sampler comparison - traces (small multiples, one sampler per panel)
# ======================================================================

def run_all(n=6000, seed=0):
    """Run every sampler on the same target, from the same bad start."""
    x0 = np.array([-3.0, 4.0])          # deliberately far from the mode
    out = {}

    t = time.perf_counter()
    out["MH"] = mc.metropolis(log_post, x0, n, proposal_cov=0.4,
                              rng=np.random.default_rng(seed))
    out["MH"].wall = time.perf_counter() - t

    t = time.perf_counter()
    out["MALA"] = mc.mala(log_post_and_grad, x0, n, step_size=0.5,
                          rng=np.random.default_rng(seed))
    out["MALA"].wall = time.perf_counter() - t

    t = time.perf_counter()
    out["RAM"] = mc.ram(log_post, x0, n, initial_cov=0.01,
                        rng=np.random.default_rng(seed))
    out["RAM"].wall = time.perf_counter() - t

    t = time.perf_counter()
    out["DRAM"] = mc.dram(log_post, x0, n, initial_cov=0.01,
                          rng=np.random.default_rng(seed))
    out["DRAM"].wall = time.perf_counter() - t

    t = time.perf_counter()
    out["AdaptiveMALA"] = mc.adaptive_mala(log_post_and_grad, x0, n,
                                           initial_step_size=0.1,
                                           rng=np.random.default_rng(seed))
    out["AdaptiveMALA"].wall = time.perf_counter() - t

    t = time.perf_counter()
    out["Gibbs"] = mc.gibbs(log_post, x0, n, proposal_std=0.8,
                            rng=np.random.default_rng(seed))
    out["Gibbs"].wall = time.perf_counter() - t

    return out


def fig_traces(results):
    fig, axes = plt.subplots(3, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    for ax, (name, res) in zip(axes.ravel(), results.items()):
        ax.plot(res.samples[:, 0], lw=0.7, color=SERIES[name])
        ax.axhline(MU[0], color=TRUTH, ls="--", lw=1.5, zorder=3)
        ax.set_title(f"{name}   (acceptance {res.acceptance_rate:.0%})",
                     loc="left")
        recessive(ax)
    for ax in axes[-1]:
        ax.set_xlabel("iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta_1$")
    fig.suptitle("Every sampler from the same bad start, dashed line = truth",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save("sampler_traces.png", fig)


def fig_efficiency(results):
    """Two panels, never a dual axis: they are different measures."""
    names = list(results)
    ess = [float(np.mean(r.ess())) / len(r.samples) * 1000 for r in results.values()]
    acc = [r.acceptance_rate * 100 for r in results.values()]
    colors = [SERIES[n] for n in names]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    for ax, vals, title, unit in (
        (a1, ess, "Effective samples per 1000 draws", ""),
        (a2, acc, "Acceptance rate", "%"),
    ):
        bars = ax.bar(names, vals, color=colors, width=0.62, zorder=2)
        # Direct labels: required relief for the low-contrast slots, and it
        # removes any need to read values off the axis.
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}{unit}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.grid(axis="x", visible=False)
        recessive(ax)

    a1.set_ylabel("effective samples")
    a2.set_ylabel("% of proposals accepted")
    fig.tight_layout()
    save("sampler_efficiency.png", fig)


# ======================================================================
# 2. A hard target: where adaptation earns its keep
# ======================================================================

def fig_banana():
    n = 20_000
    x0 = np.array([0.0, 0.0])
    runs = {
        "MH": mc.metropolis(banana, x0, n, proposal_cov=0.05,
                            rng=np.random.default_rng(1)),
        "RAM": mc.ram(banana, x0, n, initial_cov=0.05,
                      rng=np.random.default_rng(1)),
        "DRAM": mc.dram(banana, x0, n, initial_cov=0.05,
                        rng=np.random.default_rng(1)),
    }

    # Cover the range the chains actually reach, so the target density is
    # visible behind every panel rather than squashed into the bottom strip.
    xlim, ylim = (-2.6, 4.2), (-1.0, 15.0)
    gx, gy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    glogp = np.array([[banana(np.array([a, b])) for a, b in zip(rx, ry)]
                      for rx, ry in zip(gx, gy)])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)
    for ax, (name, res) in zip(axes, runs.items()):
        # Contour the log density: the linear density is so peaked that all
        # but one level would collapse onto the mode.
        ax.contour(gx, gy, glogp, levels=[-40, -20, -10, -4, -1],
                   colors="#b9b6ac", linewidths=0.9, zorder=1)
        s = res.discard(2000).samples
        ax.plot(s[:, 0], s[:, 1], ".", ms=1.2, alpha=0.25,
                color=SERIES[name], zorder=2)
        ax.set_title(f"{name}   (ESS {np.mean(res.ess()):.0f})", loc="left")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        recessive(ax)
    axes[0].set_ylabel(r"$\theta_2$")
    fig.suptitle("A curved ridge: a fixed proposal cannot fit it, an adaptive one can",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save("sampler_banana.png", fig)


# ======================================================================
# 3. RAM adapting its own scale from a terrible start
# ======================================================================

def fig_ram_adaptation():
    rng = np.random.default_rng(3)
    x = np.array([-3.0, 4.0])
    logp = log_post(x)
    S = np.linalg.cholesky(np.eye(2) * 1e-4)      # 100x too small

    n = 8000
    scale, acc_run = np.zeros(n), np.zeros(n)
    hits = 0
    for i in range(1, n + 1):
        x, logp, S, accepted = mc.ram_step(log_post, x, logp, S, i, rng=rng)
        scale[i - 1] = np.sqrt(np.trace(S @ S.T) / 2)
        hits += accepted
        acc_run[i - 1] = hits / i

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))

    a1.plot(scale, lw=2, color=SERIES["RAM"])
    a1.axhline(1.0, color=TRUTH, ls="--", lw=1.5)
    a1.text(n * 0.42, 1.08, "posterior standard deviation", color=TRUTH, fontsize=9)
    a1.set_yscale("log")
    a1.set_title("Proposal scale, started 100x too small", loc="left")
    # It settles near 2.4x the posterior sd, close to the optimal 2.38/sqrt(d)
    # scaling for a random walk.
    a1.set_xlabel("iteration")
    a1.set_ylabel("RMS proposal scale")
    recessive(a1)

    a2.plot(acc_run, lw=2, color=SERIES["RAM"])
    a2.axhline(0.234, color=TRUTH, ls="--", lw=1.5)
    a2.text(n * 0.55, 0.25, "target 0.234", color=TRUTH, fontsize=9)
    a2.set_ylim(0, 1)
    a2.set_title("Running acceptance rate", loc="left")
    a2.set_xlabel("iteration")
    a2.set_ylabel("fraction accepted")
    recessive(a2)

    fig.tight_layout()
    save("ram_adaptation.png", fig)


# ======================================================================
# 4. The plotting gallery
# ======================================================================

def fig_gallery():
    res = mc.ram(log_post, [0.0, 0.0], 20_000, initial_cov=0.1,
                 param_names=["x", "y"], rng=np.random.default_rng(7))
    res = res.discard(2000)

    res.plot_trace(title="plot_trace()")
    save("plot_trace.png")

    res.plot_marginals(title="plot_marginals()")
    save("plot_marginals.png")

    res.plot_autocorr(max_lag=60, title="plot_autocorr()")
    save("plot_autocorr.png")

    for style in ("corner", "scatter", "full", "kde"):
        fig = res.plot_corner(style=style, true_values=MU,
                              title=f'plot_corner(style="{style}")')
        save(f"corner_{style}.png", fig)

    # TMCMC stage evolution
    problem = mc.Problem(prior=lambda t: 0.0, likelihood=log_post,
                         param_names=["x", "y"])
    prior_samples = np.random.default_rng(0).uniform(-8, 8, size=(800, 2))
    tmcmc = mc.TMCMC(n_particles=800, n_mcmc_steps=3)
    tmcmc.run(problem, prior_samples=prior_samples)
    fig = tmcmc.plot_stages(max_stages=6, title="TMCMC.plot_stages()")
    save("tmcmc_stages.png", fig)

    # Posterior predictive bands
    x_grid = np.linspace(0, 10, 40)

    def forward(theta):
        return theta[0] * np.sin(0.6 * x_grid) + theta[1]

    rng = np.random.default_rng(5)
    y_obs = forward(MU) + rng.normal(0, 0.25, x_grid.size)

    def ll(theta):
        return -0.5 * np.sum(((forward(theta) - y_obs) / 0.25) ** 2)

    fit = mc.ram(ll, [1.0, 0.0], 8000, initial_cov=0.05,
                 param_names=["a", "b"], rng=np.random.default_rng(2))
    pred = fit.discard(2000).posterior_predictive(forward, n_eval=400)
    pred.plot_bands(x=x_grid, y_obs=y_obs, title="PosteriorPredictive.plot_bands()",
                    xlabel="x", ylabel="response")
    save("plot_bands.png")


if __name__ == "__main__":
    print("generating documentation figures ->", OUT)
    results = run_all()
    fig_traces(results)
    fig_efficiency(results)
    fig_banana()
    fig_ram_adaptation()
    fig_gallery()
    print("done")
