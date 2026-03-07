# Adaptive samplers

Source: [`examples/adaptive_samplers.py`](https://github.com/your-username/mcmckit/blob/main/examples/adaptive_samplers.py)

Comparison of all adaptive samplers starting from a **deliberately bad** initial proposal covariance.

---

## RAM

RAM self-corrects the proposal covariance from $0.01^2 I$ to the correct scale within a few hundred steps:

```python
ram = mc.RAM(n_samples=15_000, initial_cov=np.eye(2) * 0.01**2)
result_ram = ram.run(problem, x0=[0.0, 0.0])

print(f"final proposal std: {np.sqrt(np.diag(ram.proposal_cov))}")
```

---

## DRAM

DRAM combines adaptive covariance with delayed rejection — a second, smaller proposal is tried whenever the first is rejected:

```python
dram = mc.DRAM(n_samples=15_000, initial_cov=np.eye(2) * 0.5,
               dr_scale=0.1)
result_dram = dram.run(problem, x0=[0.0, 0.0])

print(f"stage-1 acc: {dram.stage1_acceptance_rate:.3f}")
print(f"stage-2 acc: {dram.stage2_acceptance_rate:.3f}")
```

---

## AdaptiveMALA

```python
amala = mc.AdaptiveMALA(n_samples=15_000, initial_step_size=0.05)
result_amala = amala.run(problem_grad, x0=[0.0, 0.0])

print(f"final step size: {amala.step_size:.4f}")
```

---

## Comparison table

```python
samplers = {
    "MH (tuned)":   result_mh,
    "RAM":          result_ram,
    "DRAM":         result_dram,
    "AdaptiveMALA": result_amala,
}

for name, res in samplers.items():
    print(f"{name:>14}  mean={res.mean()}  acc={res.acceptance_rate:.3f}")
```
