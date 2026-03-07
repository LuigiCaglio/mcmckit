# Result

Container for posterior samples returned by all samplers.

```python
result = sampler.run(problem, x0=[...])
result = result.discard(1000)      # remove burn-in

result.mean()                      # posterior mean
result.std()                       # posterior std
result.cov()                       # posterior covariance matrix
result.quantile([0.025, 0.975])    # credible interval
result.samples                     # raw ndarray, shape (n_samples, n_params)
result.log_posteriors              # log p(theta | y), shape (n_samples,)
result.log_evidence                # set by TMCMC; None for MCMC samplers

result.plot_trace()
result.plot_marginals()
result.plot_corner(style="corner", true_values=[...])
```

---

::: mcmckit.core.result.Result
