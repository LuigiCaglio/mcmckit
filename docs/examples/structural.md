# Structural identification

Source: [`examples/structural_identification.py`](https://github.com/your-username/mcmckit/blob/main/examples/structural_identification.py)

Identify stiffness values of a 2-DOF mass-spring system from noisy natural frequency measurements.

---

## System

```
ground ---k1--- [m1] ---k2--- [m2] ---k3--- wall
         m1 = m2 = 1.0  (known)
         k1, k2, k3    (unknown)
```

Forward model: eigenvalue problem $K\mathbf{v} = \omega^2 M\mathbf{v}$ → natural frequencies $[\omega_1, \omega_2]$.

---

## Forward model

```python
import numpy as np
import mcmckit as mc

M1, M2 = 1.0, 1.0

def natural_frequencies(k):
    k1, k2, k3 = k
    K = np.array([[k1 + k2, -k2], [-k2, k2 + k3]])
    Minvhalf = np.diag(1.0 / np.sqrt([M1, M2]))
    eigvals = np.linalg.eigvalsh(Minvhalf @ K @ Minvhalf)
    return np.sqrt(np.maximum(eigvals, 0.0))

def forward_model(theta):
    return np.tile(natural_frequencies(theta), N_repeat)
```

---

## Marginalised per-channel noise (recommended)

Different σ for each frequency — no noise parameters to sample:

```python
groups = [
    np.arange(0, N_repeat),
    np.arange(N_repeat, 2 * N_repeat),
]

ll = mc.GaussianNoiseLikelihood(
    forward_model, y_obs,
    groups=groups,
    marginalise_noise=True,
    inv_gamma_alpha=[2.0, 2.0],
    inv_gamma_beta=[0.08**2, 0.03**2],
)

def log_prior(theta):
    if np.any(theta <= 1.0) or np.any(theta >= 30.0):
        return -np.inf
    return 0.0

problem = mc.Problem(prior=log_prior, likelihood=ll,
                     param_names=["k1", "k2", "k3"])

result = mc.DRAM(n_samples=15_000,
                 initial_cov=np.diag([0.5, 0.5, 0.5])).run(
    problem, x0=[6.0, 4.0, 5.0]).discard(3000)

print(result.mean())

# posterior noise estimates per channel
sigma_est = ll.posterior_sigma(result.mean())
print(f"sigma: omega1={sigma_est[0]:.4f}, omega2={sigma_est[1]:.4f}")
```

---

## Visualisation

```python
result.plot_corner(true_values=[8.0, 5.0, 6.0],
                   title="Identified stiffness posterior")
```
