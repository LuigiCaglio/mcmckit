"""Smoke tests: verify samplers run and produce plausible output."""
import numpy as np
import pytest
import mcmckit as mc


# ---------------------------------------------------------------------------
# Shared problem: 2-D standard normal
# ---------------------------------------------------------------------------

def log_prior(theta):
    return 0.0


def log_likelihood(theta):
    return -0.5 * float(np.dot(theta, theta))


def grad_log_likelihood(theta):
    return -np.asarray(theta, dtype=float)


def grad_log_prior(theta):
    return np.zeros_like(theta)


PROBLEM = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["x", "y"],
    grad_log_likelihood=grad_log_likelihood,
    grad_log_prior=grad_log_prior,
)

X0 = [0.5, -0.5]
N = 500


# ---------------------------------------------------------------------------
# MetropolisHastings
# ---------------------------------------------------------------------------

def test_mh_run():
    sampler = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=N)
    result = sampler.run(PROBLEM, x0=X0)
    assert result.samples.shape == (N, 2)
    assert result.acceptance_rate is not None
    assert 0.0 < result.acceptance_rate < 1.0


def test_mh_step_by_step():
    sampler = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=N)
    sampler.initialize(PROBLEM, x0=X0)
    for _ in range(10):
        sampler.step()
    result = sampler.get_result()
    assert result.samples.shape == (10, 2)


# ---------------------------------------------------------------------------
# MALA
# ---------------------------------------------------------------------------

def test_mala_run():
    sampler = mc.MALA(step_size=0.3, n_samples=N)
    result = sampler.run(PROBLEM, x0=X0)
    assert result.samples.shape == (N, 2)
    assert 0.0 < result.acceptance_rate < 1.0


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def test_ram_run():
    sampler = mc.RAM(n_samples=N, initial_cov=np.eye(2) * 0.1)
    result = sampler.run(PROBLEM, x0=X0)
    assert result.samples.shape == (N, 2)
    assert sampler.proposal_cov.shape == (2, 2)


# ---------------------------------------------------------------------------
# DRAM
# ---------------------------------------------------------------------------

def test_dram_run():
    sampler = mc.DRAM(n_samples=N, initial_cov=np.eye(2) * 0.5)
    result = sampler.run(PROBLEM, x0=X0)
    assert result.samples.shape == (N, 2)
    assert 0.0 <= sampler.stage1_acceptance_rate <= 1.0


# ---------------------------------------------------------------------------
# AdaptiveMALA
# ---------------------------------------------------------------------------

def test_adaptive_mala_run():
    sampler = mc.AdaptiveMALA(n_samples=N, initial_step_size=0.1)
    result = sampler.run(PROBLEM, x0=X0)
    assert result.samples.shape == (N, 2)
    assert sampler.step_size > 0.0


# ---------------------------------------------------------------------------
# Gibbs
# ---------------------------------------------------------------------------

def test_gibbs_scalar():
    sampler = mc.Gibbs(n_samples=N, proposal_std=0.5)
    result = sampler.run(PROBLEM, x0=X0)
    assert result.samples.shape == (N, 2)
    assert len(sampler.block_acceptance_rates) == 2


def test_gibbs_block():
    sampler = mc.Gibbs(n_samples=N, blocks=[[0], [1]], proposal_std=[0.5, 0.5])
    result = sampler.run(PROBLEM, x0=X0)
    assert result.samples.shape == (N, 2)


# ---------------------------------------------------------------------------
# TMCMC
# ---------------------------------------------------------------------------

def test_tmcmc_run():
    prior_samples = np.random.uniform(-5, 5, size=(200, 2))
    tmcmc = mc.TMCMC(n_particles=200, n_mcmc_steps=2)
    result = tmcmc.run(PROBLEM, prior_samples=prior_samples)
    assert result.samples.shape == (200, 2)
    assert result.log_evidence is not None
    assert np.isfinite(result.log_evidence)
    assert tmcmc.beta == 1.0
    assert tmcmc.stage >= 1


def test_tmcmc_history():
    prior_samples = np.random.uniform(-5, 5, size=(200, 2))
    tmcmc = mc.TMCMC(n_particles=200, n_mcmc_steps=2)
    tmcmc.run(PROBLEM, prior_samples=prior_samples)
    # history starts at beta=0 and ends at beta=1
    assert tmcmc._history[0][0] == 0.0
    assert tmcmc._history[-1][0] == 1.0


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def test_result_discard():
    sampler = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=N)
    result = sampler.run(PROBLEM, x0=X0)
    trimmed = result.discard(100)
    assert trimmed.samples.shape == (N - 100, 2)


def test_result_stats():
    sampler = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=2000)
    result = sampler.run(PROBLEM, x0=X0)
    mean = result.mean()
    assert mean.shape == (2,)
    assert np.allclose(mean, [0.0, 0.0], atol=0.3)
