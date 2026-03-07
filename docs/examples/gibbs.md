# Gibbs sampling

Source: [`examples/tmcmc_and_gibbs.py`](https://github.com/your-username/mcmckit/blob/main/examples/tmcmc_and_gibbs.py)

Gibbs is useful when parameters have very different scales or are weakly coupled, and you want per-parameter tuning of the proposal.

---

## Scalar Gibbs (one parameter at a time)

```python
gibbs = mc.Gibbs(n_samples=10_000, proposal_std=0.5)
result = gibbs.run(problem, x0=[0.0, 0.0])

print(result.mean())
print(f"per-block acc: {gibbs.block_acceptance_rates}")
```

By default, each dimension gets its own scalar MH update with the same `proposal_std`.

---

## Per-block proposal std

```python
gibbs = mc.Gibbs(
    n_samples=10_000,
    blocks=[[0], [1]],          # one param per block
    proposal_std=[0.8, 0.5],    # different std per block
)
result = gibbs.run(problem, x0=[0.0, 0.0])
```

---

## Block updates

Update multiple parameters jointly per block:

```python
# 4-D problem: update params 0,1 together, then 2,3 together
gibbs = mc.Gibbs(
    n_samples=10_000,
    blocks=[[0, 1], [2, 3]],
    proposal_std=[0.5, 0.3],
)
result = gibbs.run(problem_4d, x0=[0, 0, 0, 0])
```

---

## Acceptance rates

```python
gibbs.block_acceptance_rates   # list: one rate per block
gibbs.acceptance_rate          # mean over all blocks
```

Target: 20–40% per block for scalar updates.
