# Modal likelihood (MAC)

Structural model updating when both **natural frequencies** and **mode shapes**
are available from experimental modal analysis.

## Why mode shapes?

Frequencies alone are often insufficient to uniquely identify multiple
structural parameters.  Adding mode shape information (even partial, from a
few sensor locations) greatly improves identifiability — particularly for
distinguishing stiffness distributions in different parts of the structure.

## The MAC

The **Modal Assurance Criterion** measures the correlation between two mode
shape vectors:

$$\text{MAC}(\hat{\phi}, \phi(\theta)) = \frac{\left(\hat{\phi}^T\phi(\theta)\right)^2}{\|\hat{\phi}\|^2\,\|\phi(\theta)\|^2} \in [0, 1]$$

- MAC = 1 → perfect alignment (shapes are proportional)
- MAC < 1 → shape mismatch

MAC is invariant to scaling and sign — no normalisation convention needed.

## Likelihood formulation

```python
ln p(D|θ) = Σᵣ [  -½(ω̂ᵣ/ωᵣ(θ)-1)²/σ²_freq,r   ← frequency term
                 -(1 - MACᵣ)/(2σ²_mac,r)          ← mode shape term ]
```

`sigma_mac` controls how tightly the posterior is pulled toward MAC = 1.
Typical values: **0.05–0.15**.

## Basic usage

```python
import mcmckit as mc
import numpy as np

def forward(theta):
    k1, k2 = theta
    # ... solve eigenvalue problem ...
    return freq_pred, mode_shapes_pred   # (n_modes,), (n_dof, n_modes)

ll = mc.ModalLikelihood(
    forward_model=forward,
    freq_obs=np.array([2.34, 5.87]),
    mode_shapes_obs=obs_shapes,     # shape (n_sensor_dof, 2)
    sigma_freq=0.05,
    sigma_mac=0.05,
    sensor_dofs=[0, 2, 4],          # measure only DOFs 0, 2, 4
    auto_pair=True,                 # automatic mode order matching
)

problem = mc.Problem(prior=log_prior, likelihood=ll, param_names=["k1", "k2"])
result = mc.DRAM(n_samples=20_000, initial_cov=np.eye(2)).run(problem, x0=[10.0, 8.0])
```

## Mode pairing

When `auto_pair=True` (default), the likelihood automatically assigns
predicted modes to observed modes by maximising total MAC.  This is important
because mode order can switch during sampling, and a naive index-by-index
comparison would give wrong results.

The assignment is solved with the Hungarian algorithm (O(n³)) via
`scipy.optimize.linear_sum_assignment`.  A greedy fallback is used if scipy
is not available.

## Partial instrumentation

In practice you rarely measure all DOFs.  Use `sensor_dofs` to specify which
DOF indices the sensors cover:

```python
ll = mc.ModalLikelihood(
    forward_model=full_model,   # returns full-DOF shapes
    freq_obs=freq_obs,
    mode_shapes_obs=shapes_obs,
    sigma_freq=0.02,
    sigma_mac=0.08,
    sensor_dofs=[0, 5, 10, 15],  # 4 sensors on a 20-DOF model
)
```

## Free frequency noise

Treat `sigma_freq` as an unknown by setting it to `None`.  The last
`n_modes` elements of `theta` become `[log σ_freq,1, ..., log σ_freq,nm]`:

```python
ll = mc.ModalLikelihood(
    forward_model=forward,
    freq_obs=freq_obs,
    mode_shapes_obs=shapes_obs,
    sigma_freq=None,
    sigma_mac=0.05,
)
# theta = [k1, k2, log_sigma_freq_1, log_sigma_freq_2]
print(ll.n_noise_params)   # 2
```

## Diagnostics

Check how well the posterior mean fits the data:

```python
theta_mean = result.discard(3000).mean()

# Per-mode MAC values at posterior mean
macs = ll.mac_values(theta_mean)
print(f"MAC: {macs}")       # e.g. [0.9998, 0.9994]

# Relative frequency errors (dimensionless)
errs = ll.freq_errors(theta_mean)
print(f"Δω/ω (%): {errs * 100}")

# Full MAC matrix (observed rows × predicted columns)
M = mc.mac_matrix(shapes_obs, forward(theta_mean)[1])
```

## API reference

- [`ModalLikelihood`](../api/modal.md)
- [`mac_matrix`](../api/modal.md)

See the full runnable script at `examples/modal_likelihood.py`.
