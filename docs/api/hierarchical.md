# HierarchicalProblem

Joint Bayesian inference over shared hyperparameters and group-level parameters
across J nominally identical structures (or any grouped data).

```python
hproblem = mc.HierarchicalProblem(
    hyperprior=hyperprior,          # log p(phi)
    group_prior=group_prior,        # log p(theta_j | phi)
    group_likelihoods=likelihoods,  # list of likelihoods, one per group
    n_hyper=2,
    n_group=1,
    param_names_hyper=["mu_k", "log_sigma_k"],
    param_names_group=["k"],
)

# Compatible with all samplers — run as a normal Problem
result = mc.DRAM(n_samples=20_000, initial_cov=np.eye(hproblem.n_params)).run(
    hproblem,
    x0=hproblem.default_x0(phi0=[10.0, 0.0], group_x0s=[[10.0]] * J),
)

# Extract marginal posteriors
hyper_result  = hproblem.extract_hyper(result)
group_results = [hproblem.extract_group(result, j) for j in range(J)]
```

---

::: mcmckit.core.hierarchical.HierarchicalProblem
