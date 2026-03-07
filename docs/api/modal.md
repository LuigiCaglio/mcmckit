# ModalLikelihood

Log-likelihood for Bayesian model updating with **modal data** (natural
frequencies + mode shapes).  Combines a Gaussian likelihood on frequency
residuals with a MAC-based (Modal Assurance Criterion) likelihood on mode
shapes, following the formulation of Vanik et al. (1999) and Argyris et al.
(2020).

## Likelihood formulation

For each observed mode $r$:

**Frequency term** (relative error, default):

$$\ln p(\hat{\omega}_r \mid \theta) = -\frac{1}{2}\frac{\left(\hat{\omega}_r/\omega_r(\theta) - 1\right)^2}{\sigma^2_{\text{freq},r}}$$

**Mode-shape term** (MAC-based):

$$\ln p(\hat{\phi}_r \mid \theta) = -\frac{1 - \text{MAC}_r}{2\,\sigma^2_{\text{mac},r}}$$

where the **Modal Assurance Criterion** is:

$$\text{MAC}(\hat{\phi}_r,\,\Gamma\phi_r(\theta)) = \frac{\left(\hat{\phi}_r^T\,\Gamma\phi_r(\theta)\right)^2}{\|\hat{\phi}_r\|^2\,\|\Gamma\phi_r(\theta)\|^2} \in [0,1]$$

$\Gamma$ is the sensor selection matrix (specified via `sensor_dofs`).  MAC = 1
means perfect alignment; MAC < 1 penalises mode shape mismatch.

**Total log-likelihood:**

$$\ln p(D \mid \theta) = \sum_{r=1}^{n_m} \left[\ln p(\hat{\omega}_r \mid \theta) + \ln p(\hat{\phi}_r \mid \theta)\right]$$

## Quick start

```python
import mcmckit as mc
import numpy as np

def forward(theta):
    # returns (freq_pred shape (n_modes,), mode_shapes shape (n_dof, n_modes))
    freqs, shapes = solve_eigenvalue_problem(theta)
    return freqs, shapes

ll = mc.ModalLikelihood(
    forward_model=forward,
    freq_obs=np.array([2.34, 5.87]),      # observed frequencies
    mode_shapes_obs=obs_shapes,           # shape (n_sensor_dof, 2)
    sigma_freq=0.05,                      # frequency noise std
    sigma_mac=0.05,                       # MAC sensitivity (0.05–0.20 typical)
    sensor_dofs=[0, 2, 4],                # measured DOFs (0-based)
    auto_pair=True,                       # automatic mode pairing
    freq_error="relative",                # relative frequency error
)

problem = mc.Problem(prior=log_prior, likelihood=ll, param_names=["k1", "k2"])
result = mc.DRAM(n_samples=20_000, initial_cov=np.eye(2)).run(problem, x0=[10.0, 8.0])
```

## Free frequency noise

Set `sigma_freq=None` to treat frequency noise as a free parameter.  The last
`n_modes` elements of `theta` are then interpreted as `log σ_freq,r`:

```python
ll = mc.ModalLikelihood(
    forward_model=forward,
    freq_obs=obs_freqs,
    mode_shapes_obs=obs_shapes,
    sigma_freq=None,     # free — appended to theta
    sigma_mac=0.05,
)
# theta = [k1, k2, log_sigma_freq_0, log_sigma_freq_1]
print(f"Free noise params: {ll.n_noise_params}")   # 2
```

## Diagnostic helpers

After sampling, inspect fit quality at any parameter vector:

```python
theta_mean = result.discard(3000).mean()

# MAC values per mode (1.0 = perfect alignment)
macs = ll.mac_values(theta_mean)

# Relative frequency errors (dimensionless)
errs = ll.freq_errors(theta_mean)

# Full MAC matrix between observed and predicted shapes
M = mc.mac_matrix(obs_shapes, forward(theta_mean)[1])
```

## `mac_matrix` utility

```python
M = mc.mac_matrix(shapes_a, shapes_b)
# M[i, j] = MAC(shapes_a[:, i], shapes_b[:, j])
```

Useful for cross-checking mode pairing or assessing sensor placement.

---

::: mcmckit.core.modal.ModalLikelihood

---

::: mcmckit.core.modal.mac_matrix
