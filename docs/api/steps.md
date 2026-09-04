# Step functions

Single-step sampling: you own the loop.

Every function here advances a chain by exactly one step. State goes in as
plain arguments and comes back as plain return values, so the recursion is
yours to write, stop, inspect and modify.

```python
import numpy as np
import mcmckit as mc

x = np.zeros(2)
logp = log_post(x)
S = np.linalg.cholesky(np.eye(2) * 0.1**2)

for i in range(1, 10_001):
    x, logp, S, accepted = mc.ram_step(log_post, x, logp, S, i)
```

## Sign convention

Every function takes a **log-posterior**: bigger means a better fit. If your
code carries a negative log-posterior, wrap it once:

```python
log_post = lambda theta: -my_nll(theta)
```

## Auxiliary model output

If your callable returns `(log_post, aux)`, the extra payload is threaded back
out with the accepted sample and passed back in via `aux=`. Use it to keep
whatever your forward model already computed, without re-running it.

```python
def log_post(theta):
    freqs = surrogate(theta)
    return -0.5 * np.sum(((freqs - measured) / sigma)**2), freqs

x, logp, S, accepted, freqs = mc.ram_step(log_post, x, logp, S, i, aux=freqs)
```

Return a plain float instead and no `aux` comes back, so the signature keeps
its usual width.

---

::: mcmckit.steps.mh_step

---

::: mcmckit.steps.ram_step

---

::: mcmckit.steps.dram_step

---

::: mcmckit.steps.mala_step

---

::: mcmckit.steps.adaptive_mala_step

---

::: mcmckit.steps.gibbs_step

---

::: mcmckit.steps.DRAMState

---

::: mcmckit.steps.init_dram_state

---

::: mcmckit.steps.as_cov
