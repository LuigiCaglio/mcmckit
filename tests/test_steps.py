"""Correctness tests for the single-step API.

Every step function is driven from a hand-written loop, exactly the way a
user would drive it, and must recover the mean and covariance of a known
correlated Gaussian.
"""

import numpy as np
import pytest

import mcmckit as mc


# ----------------------------------------------------------------------
# A target with an analytic answer
# ----------------------------------------------------------------------

MU = np.array([1.0, -2.0])
COV = np.array([[2.0, 0.8], [0.8, 0.5]])
COV_INV = np.linalg.inv(COV)


def log_post(theta):
    d = theta - MU
    return -0.5 * d @ COV_INV @ d


def log_post_and_grad(theta):
    d = theta - MU
    return -0.5 * d @ COV_INV @ d, -COV_INV @ d


N = 40_000
BURN = 5_000


def _check_recovers(chain):
    chain = np.asarray(chain)[BURN:]
    assert np.allclose(chain.mean(0), MU, atol=0.1), chain.mean(0)
    assert np.allclose(np.cov(chain.T), COV, atol=0.15), np.cov(chain.T)


# ----------------------------------------------------------------------
# Each step function, driven from a user-written loop
# ----------------------------------------------------------------------

def test_mh_step_recovers_gaussian():
    rng = np.random.default_rng(0)
    x, logp = np.zeros(2), log_post(np.zeros(2))
    cov = np.eye(2) * 0.5

    chain = []
    for _ in range(N):
        x, logp, _ = mc.mh_step(log_post, x, logp, cov, rng=rng)
        chain.append(x)
    _check_recovers(chain)


def test_ram_step_recovers_gaussian():
    rng = np.random.default_rng(0)
    x, logp = np.zeros(2), log_post(np.zeros(2))
    S = np.linalg.cholesky(np.eye(2) * 0.01)

    chain = []
    for i in range(1, N + 1):
        x, logp, S, _ = mc.ram_step(log_post, x, logp, S, i, rng=rng)
        chain.append(x)
    _check_recovers(chain)


def test_ram_step_adapts_from_a_bad_start():
    """S must move a long way from a deliberately terrible initial scale."""
    rng = np.random.default_rng(1)
    x, logp = np.zeros(2), log_post(np.zeros(2))
    S0 = np.linalg.cholesky(np.eye(2) * 1e-4)
    S = S0.copy()

    for i in range(1, 5_000 + 1):
        x, logp, S, _ = mc.ram_step(log_post, x, logp, S, i, rng=rng)

    assert np.trace(S @ S.T) > 10 * np.trace(S0 @ S0.T)


def test_mala_step_recovers_gaussian():
    rng = np.random.default_rng(0)
    x = np.zeros(2)
    logp, grad = log_post_and_grad(x)

    chain = []
    for _ in range(N):
        x, logp, grad, _ = mc.mala_step(
            log_post_and_grad, x, logp, grad, 0.4, rng=rng
        )
        chain.append(x)
    _check_recovers(chain)


def test_adaptive_mala_step_recovers_gaussian():
    rng = np.random.default_rng(0)
    x = np.zeros(2)
    logp, grad = log_post_and_grad(x)
    log_step = np.log(0.1)

    chain = []
    for i in range(1, N + 1):
        x, logp, grad, log_step, _ = mc.adaptive_mala_step(
            log_post_and_grad, x, logp, grad, log_step, i, rng=rng
        )
        chain.append(x)
    _check_recovers(chain)


def test_dram_step_recovers_gaussian():
    rng = np.random.default_rng(0)
    x, logp = np.zeros(2), log_post(np.zeros(2))
    state = mc.init_dram_state(x, initial_cov=0.1)

    chain = []
    for _ in range(N):
        x, logp, state, _ = mc.dram_step(log_post, x, logp, state, rng=rng)
        chain.append(x)
    _check_recovers(chain)


def test_gibbs_step_recovers_gaussian():
    rng = np.random.default_rng(0)
    x, logp = np.zeros(2), log_post(np.zeros(2))

    chain = []
    for _ in range(N):
        x, logp, acc = mc.gibbs_step(
            log_post, x, logp, blocks=[[0], [1]], proposal_std=1.0, rng=rng
        )
        assert len(acc) == 2
        chain.append(x)
    _check_recovers(chain)


# ----------------------------------------------------------------------
# State threading
# ----------------------------------------------------------------------

def test_dram_recursive_covariance_matches_full_history():
    """The Welford update must equal np.cov over the whole chain."""
    rng = np.random.default_rng(3)
    x, logp = np.zeros(2), log_post(np.zeros(2))
    state = mc.init_dram_state(x, initial_cov=0.1)

    visited = []
    for _ in range(500):
        x, logp, state, _ = mc.dram_step(log_post, x, logp, state, rng=rng)
        visited.append(x.copy())

    visited = np.array(visited)
    assert state.n == 500
    assert np.allclose(state.mean, visited.mean(0))
    assert np.allclose(state.M2 / (state.n - 1), np.cov(visited.T))


def test_dram_state_is_plain_data():
    state = mc.init_dram_state(np.zeros(2))
    C, mean, M2, n = state          # unpacks like a tuple
    assert n == 0
    assert C.shape == (2, 2)
    assert isinstance(state, mc.DRAMState)


def test_step_index_must_be_one_indexed():
    x, logp = np.zeros(2), log_post(np.zeros(2))
    S = np.linalg.cholesky(np.eye(2) * 0.1)
    with pytest.raises(ValueError, match="1-indexed"):
        mc.ram_step(log_post, x, logp, S, 0)


def test_seeded_runs_are_reproducible():
    def run(seed):
        rng = np.random.default_rng(seed)
        x, logp = np.zeros(2), log_post(np.zeros(2))
        S = np.linalg.cholesky(np.eye(2) * 0.1)
        out = []
        for i in range(1, 201):
            x, logp, S, _ = mc.ram_step(log_post, x, logp, S, i, rng=rng)
            out.append(x.copy())
        return np.array(out)

    assert np.array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))


# ----------------------------------------------------------------------
# Auxiliary model output passthrough
# ----------------------------------------------------------------------

def log_post_with_aux(theta):
    """Mimics a forward model that also returns something you want to keep."""
    d = theta - MU
    freqs = np.array([theta[0] * 2.0, theta[1] * 3.0])
    return -0.5 * d @ COV_INV @ d, freqs


def test_aux_is_threaded_through_mh():
    rng = np.random.default_rng(0)
    x = np.zeros(2)
    logp, aux = log_post_with_aux(x)

    for _ in range(200):
        x, logp, accepted, aux = mc.mh_step(
            log_post_with_aux, x, logp, np.eye(2) * 0.3, aux=aux, rng=rng
        )
        # aux must always correspond to the returned position
        assert np.allclose(aux, [x[0] * 2.0, x[1] * 3.0])


def test_aux_is_threaded_through_ram():
    rng = np.random.default_rng(0)
    x = np.zeros(2)
    logp, aux = log_post_with_aux(x)
    S = np.linalg.cholesky(np.eye(2) * 0.1)

    for i in range(1, 201):
        x, logp, S, accepted, aux = mc.ram_step(
            log_post_with_aux, x, logp, S, i, aux=aux, rng=rng
        )
        assert np.allclose(aux, [x[0] * 2.0, x[1] * 3.0])


def test_aux_is_threaded_through_dram():
    rng = np.random.default_rng(0)
    x = np.zeros(2)
    logp, aux = log_post_with_aux(x)
    state = mc.init_dram_state(x, 0.1)

    for _ in range(200):
        x, logp, state, accepted, aux = mc.dram_step(
            log_post_with_aux, x, logp, state, aux=aux, rng=rng
        )
        assert np.allclose(aux, [x[0] * 2.0, x[1] * 3.0])


def test_no_aux_keeps_the_signature_narrow():
    x, logp = np.zeros(2), log_post(np.zeros(2))
    out = mc.mh_step(log_post, x, logp, np.eye(2) * 0.1)
    assert len(out) == 3


def test_bad_aux_tuple_is_rejected():
    def three_tuple(theta):
        return 0.0, 1, 2

    x = np.zeros(2)
    with pytest.raises(ValueError, match="2-tuple"):
        mc.mh_step(three_tuple, x, 0.0, np.eye(2) * 0.1)


# ----------------------------------------------------------------------
# The full-run helpers are genuinely the same code
# ----------------------------------------------------------------------

def test_runner_matches_hand_written_loop():
    """mc.ram(...) must equal the user's own loop over ram_step, same seed."""
    result = mc.ram(log_post, x0=np.zeros(2), n_samples=300,
                    initial_cov=0.1, rng=np.random.default_rng(11))

    rng = np.random.default_rng(11)
    x, logp = np.zeros(2), log_post(np.zeros(2))
    S = np.linalg.cholesky(mc.as_cov(0.1, 2))
    by_hand = []
    for i in range(1, 301):
        x, logp, S, _ = mc.ram_step(log_post, x, logp, S, i, rng=rng)
        by_hand.append(x.copy())

    assert np.allclose(result.samples, np.array(by_hand))


def test_runners_recover_gaussian():
    for name, kwargs in [
        ("metropolis", dict(proposal_cov=0.5)),
        ("ram", dict(initial_cov=0.01)),
        ("dram", dict(initial_cov=0.1)),
        ("gibbs", dict(proposal_std=1.0)),
    ]:
        runner = getattr(mc, name)
        res = runner(log_post, x0=np.zeros(2), n_samples=N,
                     rng=np.random.default_rng(0), **kwargs)
        _check_recovers(res.samples)


def test_runners_accept_a_problem_object():
    problem = mc.Problem(prior=lambda t: 0.0, likelihood=log_post,
                         param_names=["a", "b"])
    res = mc.ram(problem, x0=np.zeros(2), n_samples=2000,
                 rng=np.random.default_rng(0))
    assert res.param_names == ["a", "b"]


def test_gibbs_runner_reports_per_block_rates():
    res = mc.gibbs(log_post, x0=np.zeros(2), n_samples=2000,
                   blocks=[[0], [1]], proposal_std=1.0,
                   rng=np.random.default_rng(0))
    assert len(res.block_acceptance_rates) == 2
    assert np.all(res.block_acceptance_rates > 0)
