# GaussianNoiseLikelihood

Wraps a forward model and observed data into a log-likelihood for Gaussian measurement noise.

Supports three modes:

| Mode | `noise_std` | `marginalise_noise` | Noise in `theta`? |
|---|---|---|---|
| Fixed σ | float or array | — | No |
| Estimated σ | `None` | `False` | Yes (`log_σ` appended) |
| Marginalised σ | `None` | `True` | No |

Per-channel noise is supported via `groups`.

```python
import mcmckit as mc

# Fixed scalar noise
ll = mc.GaussianNoiseLikelihood(forward_model, y_obs, noise_std=0.05)

# Estimated noise (last element of theta = log_sigma)
ll = mc.GaussianNoiseLikelihood(forward_model, y_obs)

# Marginalised noise — no noise parameter in theta
ll = mc.GaussianNoiseLikelihood(
    forward_model, y_obs,
    marginalise_noise=True,
    inv_gamma_alpha=2.0,
    inv_gamma_beta=0.05**2,
)

# Per-channel (different sigma per measurement type)
ll = mc.GaussianNoiseLikelihood(
    forward_model, y_obs,
    groups=[[0,1,2], [3,4,5]],
    marginalise_noise=True,
    inv_gamma_alpha=[2.0, 2.0],
    inv_gamma_beta=[0.05**2, 0.10**2],
)
```

---

::: mcmckit.core.noise.GaussianNoiseLikelihood
