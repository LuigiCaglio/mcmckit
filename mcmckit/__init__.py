"""mcmckit - minimal, plug-and-play MCMC samplers.

Two ways in, running identical code underneath.

**You own the loop** (``mcmckit.steps``). One step at a time, state threaded
in and out as plain values::

    x, logp, S, accepted = mc.ram_step(log_post, x, logp, S, i)

**Or hand over the loop** (``mcmckit.runners``)::

    result = mc.ram(log_post, x0, n_samples=10_000)
"""

# --- step functions: you own the recursion -----------------------------
from .steps import (
    mh_step,
    mala_step,
    ram_step,
    adaptive_mala_step,
    dram_step,
    gibbs_step,
    DRAMState,
    init_dram_state,
    as_cov,
)

# --- full-run helpers: thin loops over the step functions --------------
from .runners import metropolis, ram, mala, adaptive_mala, dram, gibbs

# --- problem / result containers ---------------------------------------
from .core.problem import Problem
from .core.result import Result, PosteriorPredictive
from .core.noise import GaussianNoiseLikelihood
from .core.similarity import mac, mac_matrix
from .core.hierarchical import HierarchicalProblem
from .core.model_comparison import bayes_factor, ModelComparison, BMAResult
from .core.diagnostics import ess, autocorr, gelman_rubin, convergence_summary
from .core.multichain import MultiChainResult, run_chains
from .core.sequential import PosteriorPrior
from .core.parallel import WorkerPool

# --- sampler classes: stateful wrappers, for stop-and-inspect workflows -
from .samplers.metropolis import MetropolisHastings
from .samplers.mala import MALA
from .samplers.ram import RAM
from .samplers.dram import DRAM
from .samplers.adaptive_mala import AdaptiveMALA
from .samplers.tmcmc import TMCMC
from .samplers.gibbs import Gibbs

__version__ = "0.3.0"

__all__ = [
    # step functions - you own the loop
    "mh_step", "mala_step", "ram_step", "adaptive_mala_step",
    "dram_step", "gibbs_step",
    "DRAMState", "init_dram_state", "as_cov",
    # full-run helpers
    "metropolis", "ram", "mala", "adaptive_mala", "dram", "gibbs",
    # containers and utilities
    "Problem", "Result", "PosteriorPredictive",
    "GaussianNoiseLikelihood", "HierarchicalProblem",
    "mac", "mac_matrix",
    "bayes_factor", "ModelComparison", "BMAResult",
    "ess", "autocorr", "gelman_rubin", "convergence_summary",
    "MultiChainResult", "run_chains", "PosteriorPrior",
    "WorkerPool",
    # sampler classes
    "MetropolisHastings", "MALA", "RAM", "DRAM", "AdaptiveMALA", "TMCMC", "Gibbs",
    "__version__",
]
