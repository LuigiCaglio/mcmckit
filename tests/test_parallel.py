"""Parallel evaluation: opt-in, and it must not change the answer.

Everything at module level here is deliberate - process workers receive their
work by pickling it, and anything defined inside a function or fixture cannot be
pickled.
"""

import numpy as np
import pytest

import mcmckit as mc
from mcmckit.core.parallel import WorkerPool, check_picklable, resolve_n_workers

# ---------------------------------------------------------------------------
# Module-level problem, so the process backend can pickle it
# ---------------------------------------------------------------------------

TAU, SIGMA_OBS = 3.0, 1.0
Y_OBS = np.array([1.0, -0.5])


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


def square(x):
    return x * x


PROBLEM = mc.Problem(prior=log_prior, likelihood=log_likelihood, param_names=["a", "b"])


# ---------------------------------------------------------------------------
# resolve_n_workers / check_picklable
# ---------------------------------------------------------------------------

def test_resolve_n_workers():
    import os

    assert resolve_n_workers(None) == 1
    assert resolve_n_workers(1) == 1
    assert resolve_n_workers(4) == 4
    assert resolve_n_workers(-1) == (os.cpu_count() or 1)


@pytest.mark.parametrize("bad", [0, -2, -10])
def test_resolve_n_workers_rejects_nonsense(bad):
    with pytest.raises(ValueError, match="n_workers"):
        resolve_n_workers(bad)


def test_check_picklable_distinguishes_lambdas_from_module_functions():
    assert check_picklable(log_likelihood) is True
    assert check_picklable(lambda x: x) is False


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def test_single_worker_creates_no_pool():
    pool = WorkerPool(n_workers=1, backend="auto", func=log_likelihood)
    assert pool.backend == "serial"
    with pool:
        assert pool._executor is None
        assert pool.map(square, [1, 2, 3]) == [1, 4, 9]


def test_auto_picks_processes_for_a_picklable_likelihood():
    assert WorkerPool(n_workers=2, backend="auto", func=log_likelihood).backend == "process"


def test_auto_falls_back_to_threads_for_a_lambda():
    """A lambda cannot be pickled, so 'auto' must not choose processes."""
    assert WorkerPool(n_workers=2, backend="auto", func=lambda t: 0.0).backend == "thread"


def test_explicit_process_backend_rejects_an_unpicklable_likelihood():
    """The raw pickling error is unhelpful, so this must fail early and clearly."""
    with pytest.raises(ValueError, match="picklable"):
        WorkerPool(n_workers=2, backend="process", func=lambda t: 0.0)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be"):
        WorkerPool(n_workers=2, backend="cluster", func=log_likelihood)


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_pool_map_matches_serial(backend):
    items = list(range(20))
    expected = [square(x) for x in items]
    with WorkerPool(n_workers=2, backend=backend, func=square) as pool:
        assert pool.map(square, items) == expected


# ---------------------------------------------------------------------------
# The samplers must give the same answers with workers as without
# ---------------------------------------------------------------------------

def test_tmcmc_parallel_matches_serial():
    """Parallelism spreads likelihood calls over cores; it must not change results."""
    np.random.seed(0)
    prior_samples = np.random.normal(0.0, TAU, size=(200, 2))

    np.random.seed(7)
    serial = mc.TMCMC(n_particles=200, n_mcmc_steps=3, n_workers=1).run(
        PROBLEM, prior_samples=prior_samples
    )
    np.random.seed(7)
    parallel = mc.TMCMC(n_particles=200, n_mcmc_steps=3, n_workers=2).run(
        PROBLEM, prior_samples=prior_samples
    )

    assert np.allclose(serial.samples, parallel.samples)
    assert np.isclose(serial.log_evidence, parallel.log_evidence)


def test_tmcmc_defaults_to_serial():
    assert mc.TMCMC(n_particles=10).n_workers == 1


def test_tmcmc_run_stage_still_works_outside_the_pool():
    """Driving stages by hand bypasses run(), so it must fall back to serial."""
    np.random.seed(0)
    prior_samples = np.random.normal(0.0, TAU, size=(100, 2))
    sampler = mc.TMCMC(n_particles=100, n_mcmc_steps=2, n_workers=4)
    sampler.initialize_with_samples(PROBLEM, prior_samples)
    sampler.run_stage()
    result = sampler.get_result()
    assert result.samples.shape == (100, 2)


def test_run_chains_parallel_returns_every_chain():
    np.random.seed(0)
    multi = mc.run_chains(
        mc.MetropolisHastings(proposal_cov=np.eye(2), n_samples=300),
        PROBLEM,
        x0=[0.0, 0.0],
        n_chains=4,
        n_workers=2,
    )
    assert len(multi) == 4
    for chain in multi:
        assert chain.samples.shape == (300, 2)


def test_run_chains_does_not_spawn_more_workers_than_chains():
    """Two chains asked to run on eight workers should not create six idle ones."""
    np.random.seed(0)
    multi = mc.run_chains(
        mc.MetropolisHastings(proposal_cov=np.eye(2), n_samples=100),
        PROBLEM,
        x0=[0.0, 0.0],
        n_chains=2,
        n_workers=8,
    )
    assert len(multi) == 2


def test_run_chains_defaults_to_serial_and_stays_reproducible():
    def run():
        np.random.seed(99)
        return mc.run_chains(
            mc.MetropolisHastings(proposal_cov=np.eye(2), n_samples=200),
            PROBLEM,
            x0=[0.0, 0.0],
            n_chains=2,
        )

    a, b = run(), run()
    for chain_a, chain_b in zip(a, b):
        assert np.array_equal(chain_a.samples, chain_b.samples)
