# Your own loop

Source: [`examples/own_loop.py`](https://github.com/LuigiCaglio/mcmckit/blob/master/examples/own_loop.py)

The interface mcmckit is built around. A 3-storey shear building, with the
lower two storeys identified from measured natural frequencies, sampled from a
loop you write yourself.

## The forward model

Natural frequencies from an eigenvalue problem, with the top storey known.
Identifying all three storeys from three frequencies is not identifiable:
several stiffness combinations reproduce the same spectrum to within
measurement noise.

```python
def natural_frequencies(stiffness_factors):
    k = K_NOMINAL * np.asarray(stiffness_factors, dtype=float)
    K = assemble_shear_stiffness(k)
    M = np.eye(N_DOF) * M_STOREY
    eigenvalues = np.linalg.eigvalsh(np.linalg.solve(M, K))
    return np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
```

## The log posterior, carrying its model output

Returning a tuple opts into auxiliary passthrough, so the frequencies ride
along with each accepted sample and the posterior predictive costs nothing
extra.

```python
def log_post(theta):
    if np.any(theta <= 0.1) or np.any(theta > 2.0):     # uniform prior box
        return -np.inf, np.full(N_DOF, np.nan)

    freqs = natural_frequencies([*theta, K3_KNOWN])
    log_lik = -0.5 * np.sum(((freqs - MEASURED) / SIGMA) ** 2)
    return log_lik, freqs
```

## The loop

Every piece of state is explicit: position, density, adaptation factor, step
index. Storage and stopping rules are yours.

```python
x = np.ones(n_par)
logp, freqs = log_post(x)
S = np.linalg.cholesky(np.eye(n_par) * 0.05**2)
rng = np.random.default_rng(42)

chain = np.zeros((n_iter, n_par))
freq_history = np.zeros((n_iter, N_DOF))
n_accepted = 0

for i in range(1, n_iter + 1):
    x, logp, S, accepted, freqs = mc.ram_step(
        log_post, x, logp, S, i, aux=freqs, rng=rng
    )

    chain[i - 1] = x
    freq_history[i - 1] = freqs        # kept for free, no extra model calls
    n_accepted += accepted

    if i % 5_000 == 0:                 # your own convergence check
        recent = chain[max(0, i - 5_000):i]
        if i >= 15_000 and recent.std(0).max() < 1e-4:
            chain = chain[:i]
            break
```

## Output

```
true stiffness factors : [1.  0.7]  (k3 known = 1.0)
measured frequencies   : [ 2.956  7.362 10.754] Hz

  step   5000  acc=0.22  mean=[0.999 0.703]  scale=[0.0197 0.0178]
  step  10000  acc=0.23  mean=[0.999 0.703]  scale=[0.0186 0.0188]
  step  15000  acc=0.23  mean=[1.    0.702]  scale=[0.0188 0.0187]
  step  20000  acc=0.23  mean=[0.999 0.703]  scale=[0.0199 0.0191]

posterior mean : [0.999 0.703]   (true [1.  0.7])
posterior std  : [0.008 0.008]
acceptance     : 0.23

posterior predictive frequencies [Hz]
  mode 1:  2.952 +/- 0.018   (measured  2.956)
  mode 2:  7.369 +/- 0.014   (measured  7.362)
  mode 3: 10.746 +/- 0.015   (measured 10.754)
```

RAM converges to the target 0.234 acceptance rate on its own, from a rough
starting scale.

## Plotting a chain you built yourself

A raw array wraps straight into `Result` when you want the plots:

```python
mc.Result(samples=posterior, param_names=["k1", "k2"]).plot_corner(
    true_values=TRUE_PARAMS, title="Posterior"
)
```

## The same run, handing the loop over

```python
result = mc.ram(
    lambda th: log_post(th)[0],
    x0=np.ones(n_par), n_samples=n_iter,
    initial_cov=0.05**2, param_names=["k1", "k2"],
    rng=np.random.default_rng(42),
)
```

Same seed, same numbers: the helper is a loop over the same `ram_step`.
