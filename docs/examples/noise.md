# Noise estimation

Source: [`examples/noise_estimation.py`](https://github.com/LuigiCaglio/mcmckit/blob/master/examples/noise_estimation.py)

`GaussianNoiseLikelihood` wraps a forward model and observed data into a log-likelihood function, handling measurement noise in three ways.

---

## Fixed noise

Use when σ is known from sensor specs or prior calibration:

```python
import numpy as np
import mcmckit as mc

def forward_model(theta):
    k, c = theta
    return k * np.exp(-c * x_input)

ll = mc.GaussianNoiseLikelihood(forward_model, y_obs, noise_std=0.05)

problem = mc.Problem(prior=log_prior, likelihood=ll,
                     param_names=["k", "c"])
```

---

## Estimated noise

When σ is unknown, treat `log_sigma` as the last element of `theta`:

```python
ll = mc.GaussianNoiseLikelihood(forward_model, y_obs)   # noise_std=None

problem = mc.Problem(prior=log_prior_with_sigma, likelihood=ll,
                     param_names=["k", "c", "log_sigma"])

# theta = [k, c, log_sigma] — all sampled jointly
result = mc.DRAM(n_samples=10_000, initial_cov=np.diag([0.1, 0.02, 0.1])).run(
    problem, x0=[2.0, 0.3, np.log(0.1)])

sigma_samples = np.exp(result.samples[:, 2])
print(f"sigma posterior mean: {sigma_samples.mean():.4f}")
```

The prior on `log_sigma` must be supplied by the user.  A weakly informative choice is a half-normal on σ:

```python
def log_prior_with_sigma(theta):
    k, c, log_s = theta
    if k <= 0 or c <= 0:
        return -np.inf
    sigma = np.exp(log_s)
    return -0.5 * sigma**2    # HalfNormal(1) on sigma
```

---

## Marginalised noise

Integrates σ² out analytically using an Inverse-Gamma prior.  **No noise parameter in theta** — works for any nonlinear forward model:

```python
ll = mc.GaussianNoiseLikelihood(
    forward_model, y_obs,
    marginalise_noise=True,
    inv_gamma_alpha=2.0,
    inv_gamma_beta=0.05**2,     # encodes prior belief σ ≈ 0.05
)

problem = mc.Problem(prior=log_prior, likelihood=ll,
                     param_names=["k", "c"])

result = mc.DRAM(n_samples=10_000, initial_cov=np.diag([0.1, 0.02])).run(
    problem, x0=[2.0, 0.3])

# posterior mean of sigma at the MAP
sigma_est = ll.posterior_sigma(result.mean())
print(f"sigma estimate: {float(sigma_est):.4f}")
```

---

## Per-channel noise

For multiple measurement types with different noise levels:

```python
# y_obs = [omega1_meas × N_repeat, omega2_meas × N_repeat]
groups = [
    np.arange(0, N_repeat),
    np.arange(N_repeat, 2 * N_repeat),
]

ll = mc.GaussianNoiseLikelihood(
    forward_model, y_obs,
    groups=groups,
    marginalise_noise=True,
    inv_gamma_alpha=[2.0, 2.0],
    inv_gamma_beta=[0.08**2, 0.03**2],   # different prior per channel
)

sigma_est = ll.posterior_sigma(result.mean())  # shape (2,)
print(f"sigma per channel: {sigma_est}")
```

See [Structural identification](structural.md) for a complete engineering example.
