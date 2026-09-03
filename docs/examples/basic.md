# Basic MH & MALA

Source: [`examples/simple_gaussian.py`](https://github.com/LuigiCaglio/mcmckit/blob/master/examples/simple_gaussian.py) and [`examples/mala_vs_mh.py`](https://github.com/LuigiCaglio/mcmckit/blob/master/examples/mala_vs_mh.py)

---

## Problem setup

A 2-D correlated Gaussian posterior — a simple testbed that has a known analytical solution:

```python
import numpy as np
import mcmckit as mc

true_mean = np.array([2.0, -1.0])
true_cov  = np.array([[1.0, 0.8], [0.8, 1.0]])
true_prec = np.linalg.inv(true_cov)

def log_prior(theta):
    return 0.0   # flat

def log_likelihood(theta):
    diff = theta - true_mean
    return -0.5 * diff @ true_prec @ diff

problem = mc.Problem(prior=log_prior, likelihood=log_likelihood,
                     param_names=["x", "y"])
```

---

## Metropolis-Hastings

```python
mh = mc.MetropolisHastings(proposal_cov=np.eye(2) * 0.5, n_samples=10_000)
result = mh.run(problem, x0=[0.0, 0.0])
result = result.discard(1000)   # discard burn-in

print(result.mean())            # → [~2.0, ~-1.0]
print(f"acceptance rate: {result.acceptance_rate:.3f}")
```

Access raw samples at any time:

```python
samples = result.samples         # (9000, 2) ndarray
x_chain = result.samples[:, 0]  # first parameter chain
```

---

## MALA

MALA requires gradients. Provide them on the `Problem`:

```python
def grad_log_likelihood(theta):
    return -true_prec @ (theta - true_mean)

def grad_log_prior(theta):
    return np.zeros_like(theta)

problem_grad = mc.Problem(
    prior=log_prior,
    likelihood=log_likelihood,
    grad_log_likelihood=grad_log_likelihood,
    grad_log_prior=grad_log_prior,
    param_names=["x", "y"],
)

mala = mc.MALA(step_size=0.5, n_samples=10_000)
result_mala = mala.run(problem_grad, x0=[0.0, 0.0])
```

MALA typically achieves a higher effective sample size per function evaluation than MH on smooth posteriors.

---

## Corner plot styles

```python
result.plot_corner(true_values=true_mean, style="corner")   # default: KDE contours
result.plot_corner(true_values=true_mean, style="scatter")  # scatter points
result.plot_corner(true_values=true_mean, style="full")     # scatter + KDE upper
result.plot_corner(true_values=true_mean, style="kde")      # KDE everywhere
```
