# Problem

Defines the Bayesian inference problem by wrapping the prior and likelihood as plain Python callables.

```python
import mcmckit as mc

problem = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    param_names=["E", "zeta"],
    grad_log_likelihood=grad_ll,   # optional, required for MALA
    grad_log_prior=grad_lp,        # optional, required for MALA
)
```

---

::: mcmckit.core.problem.Problem
