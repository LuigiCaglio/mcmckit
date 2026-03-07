# Result

Container for posterior samples returned by all samplers.

```python
result = sampler.run(problem, x0=[...])
result = result.discard(1000)      # remove burn-in

# Summary statistics
result.mean()                      # posterior mean, shape (n_params,)
result.std()                       # posterior std, shape (n_params,)
result.cov()                       # posterior covariance matrix
result.quantile([0.025, 0.975])    # credible interval

# Raw data
result.samples                     # ndarray, shape (n_samples, n_params)
result.log_posteriors              # log p(theta | y), shape (n_samples,)
result.log_evidence                # set by TMCMC; None for MCMC samplers

# Diagnostics
result.ess()                       # effective sample size, shape (n_params,)
result.autocorr(max_lag=100)       # ACF, shape (max_lag+1, n_params)

# Posterior predictive
pp = result.posterior_predictive(forward_model, n_eval=500)
pp.mean()                          # shape (n_obs,)
pp.plot_bands(x=..., y_obs=...)

# Plots
result.plot_trace()
result.plot_marginals()
result.plot_corner(style="corner", true_values=[...])
result.plot_autocorr(max_lag=80)
```

---

::: mcmckit.core.result.Result
