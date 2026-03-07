# Theory

## Bayesian model updating

Given a model with parameters $\theta$ and observed data $y$, Bayes' theorem gives the posterior:

$$p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}$$

where:

- $p(\theta)$ is the **prior** — knowledge about $\theta$ before seeing data.
- $p(y \mid \theta)$ is the **likelihood** — how probable the data is given $\theta$.
- $p(y) = \int p(y \mid \theta)\, p(\theta)\, d\theta$ is the **evidence** (marginal likelihood).

All samplers in mcmckit work with the **log-posterior**:

$$\log p(\theta \mid y) = \log p(y \mid \theta) + \log p(\theta) + \text{const}$$

---

## Metropolis-Hastings

At each step, a proposal $\theta^* \sim q(\cdot \mid \theta_t)$ is accepted with probability:

$$\alpha = \min\!\left(1,\ \frac{p(\theta^* \mid y)\, q(\theta_t \mid \theta^*)}{p(\theta_t \mid y)\, q(\theta^* \mid \theta_t)}\right)$$

For a symmetric Gaussian proposal $q(\theta^* \mid \theta_t) = \mathcal{N}(\theta_t, \Sigma)$ this simplifies to:

$$\alpha = \min\!\left(1,\ \exp\!\bigl[\log p(\theta^*\mid y) - \log p(\theta_t\mid y)\bigr]\right)$$

**Optimal acceptance rate** for a $d$-dimensional Gaussian target is approximately 23.4%.

---

## MALA — Metropolis-adjusted Langevin Algorithm

MALA adds a gradient drift to the proposal:

$$\theta^* = \theta_t + \tfrac{\varepsilon^2}{2}\, \nabla\!\log p(\theta_t \mid y) + \varepsilon\, \xi, \quad \xi \sim \mathcal{N}(0, I)$$

The gradient bias makes proposals more likely to move towards high-density regions. Because the proposal is asymmetric, the MH correction uses the full ratio:

$$\alpha = \min\!\left(1,\ \frac{p(\theta^* \mid y)\, q(\theta_t \mid \theta^*)}{p(\theta_t \mid y)\, q(\theta^* \mid \theta_t)}\right)$$

**Optimal acceptance rate** for MALA is approximately 57.4%.

---

## Adaptive algorithms

### RAM — Robust Adaptive Metropolis

RAM (Vihola 2012) adapts the Cholesky factor $S_t$ of the proposal covariance via a rank-1 update targeting a desired acceptance rate $\alpha^*$:

$$S_{t+1} S_{t+1}^T = S_t \!\left(I + \eta_t (\alpha_t - \alpha^*)\, \frac{r_t r_t^T}{\|r_t\|^2}\right) S_t^T$$

where $r_t = \theta^* - \theta_t$ and $\eta_t = t^{-\gamma}$ (default $\gamma = 0.51$). The target rate is $\alpha^* = 0.234$.

### DRAM — Delayed Rejection Adaptive Metropolis

DRAM (Haario et al. 2006) combines two ideas:

**Adaptive Metropolis (AM):** Update the proposal covariance using the empirical covariance of the chain so far:

$$\Sigma_t = s_d \cdot \text{Cov}(\theta_1, \ldots, \theta_t) + s_d \varepsilon I$$

**Delayed Rejection (DR):** If the first proposal $\theta^*_1$ is rejected, a second proposal $\theta^*_2$ is drawn from a smaller proposal and accepted with a corrected probability (Tierney-Mira correction) that maintains detailed balance.

### AdaptiveMALA

Adapts the MALA step size $\varepsilon$ in log-space:

$$\log \varepsilon_{t+1} = \log \varepsilon_t + \eta_t\, (\alpha_t - 0.574)$$

where $\eta_t = t^{-\gamma}$ (default $\gamma = 0.6$).

---

## TMCMC — Transitional Markov Chain Monte Carlo

TMCMC (Ching & Chen 2007) bridges the prior to the posterior through a sequence of tempered intermediate distributions:

$$p_j(\theta) \propto p(y \mid \theta)^{\beta_j}\, p(\theta), \quad 0 = \beta_0 < \beta_1 < \cdots < \beta_J = 1$$

At each stage $j$:

1. **Importance weights:** $w_j(\theta) \propto p(y\mid\theta)^{\Delta\beta_j}$, $\Delta\beta_j = \beta_j - \beta_{j-1}$
2. **Log-evidence contribution:**

$$\log p(y) \mathrel{+}= \log \mathbb{E}\!\left[p(y\mid\theta)^{\Delta\beta_j}\right]$$

computed in a numerically stable way as $\log(\text{mean}(\exp(\Delta\beta_j \cdot \ell_i - \Delta\beta_j \cdot \max \ell)))+ \Delta\beta_j \cdot \max \ell$.

3. **Systematic resampling** of particles according to $w_j$.
4. **MH rejuvenation** using the weighted covariance as proposal.

**Adaptive $\beta$ schedule:** at each stage, $\Delta\beta$ is chosen by bisection so that the effective sample size ratio $\text{ESS}/N \approx \tau$ (default $\tau = 0.5$).

---

## Gibbs sampling

Metropolis-within-Gibbs updates one block of parameters at a time, holding the rest fixed. For block $k$:

$$\theta^{(k)*} = \theta^{(k)}_t + \sigma_k\, \xi, \quad \xi \sim \mathcal{N}(0, I_{|k|})$$

accepted with probability $\min(1, \exp[\log p(\theta^* \mid y) - \log p(\theta_t \mid y)])$.

This preserves the correct joint distribution even though only one block moves per step.

---

## Gaussian noise model

For a forward model $f(\theta)$ and observed data $y$, the Gaussian likelihood is:

$$\log p(y \mid \theta, \sigma) = -\frac{1}{2\sigma^2}\|y - f(\theta)\|^2 - n\log\sigma$$

### Estimated $\sigma$

The last element of $\theta$ is $\log\sigma$ (log-transform ensures positivity) and is sampled jointly.

### Marginalised $\sigma$

With an Inverse-Gamma$(\alpha, \beta)$ prior on $\sigma^2$, integration over $\sigma^2$ is analytic for any $f(\theta)$:

$$\log p(y \mid \theta) = -\!\left(\alpha + \tfrac{n}{2}\right)\log\!\left(\beta + \tfrac{1}{2}\|y - f(\theta)\|^2\right) + \text{const}$$

The posterior mean of $\sigma$ is $\sqrt{(\beta + \text{RSS}/2)\,/\,(\alpha + n/2 - 1)}$.

### Per-channel noise

When different measurement types have different noise levels (e.g. frequency 1 vs frequency 2), observations are partitioned into channels. Each channel $i$ has its own $\sigma_i$ and the total log-likelihood is:

$$\log p(y \mid \theta) = \sum_i \log p(y_i \mid \theta, \sigma_i)$$

In the marginalised case, each channel has an independent InvGamma prior and is integrated separately.

---

## Convergence diagnostics

### Effective sample size

MCMC samples are correlated, so $N$ samples are worth fewer than $N$
independent draws.  The **effective sample size** (ESS) quantifies this:

$$\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

where $\rho_k$ is the autocorrelation at lag $k$.  The sum is truncated using
Geyer's (1992) initial positive sequence criterion to avoid noise blow-up at
large lags.

### Autocorrelation

The normalised autocorrelation at lag $k$ for a scalar chain $\{\theta_t\}$ is:

$$\rho_k = \frac{\sum_{t=1}^{N-k}(\theta_t - \bar\theta)(\theta_{t+k} - \bar\theta)}{\sum_{t=1}^{N}(\theta_t - \bar\theta)^2}$$

Computed via FFT for efficiency.  Slow decay (high autocorrelation at large lags)
indicates poor mixing — increase the proposal covariance or run longer.

### Gelman-Rubin $\hat{R}$

Given $M$ independent chains of length $N$, split each chain in half to obtain
$2M$ sub-chains.  Define the between-chain variance $B$ and within-chain
variance $W$:

$$B = \frac{N}{2M-1}\sum_{m=1}^{2M}(\bar\theta_m - \bar\theta)^2, \qquad
  W = \frac{1}{2M}\sum_{m=1}^{2M} s_m^2$$

The pooled variance estimate is:

$$\widehat{\text{var}} = \frac{N-1}{N} W + \frac{B}{N}$$

and the statistic is:

$$\hat{R} = \sqrt{\frac{\widehat{\text{var}}}{W}}$$

$\hat{R} \approx 1$ means all chains sample the same distribution.
A common threshold is $\hat{R} < 1.01$.  The split-chain variant (Vehtari
et al., 2021) also detects non-stationarity within a single chain.

---

## Model class selection

### The evidence as a model score

The **evidence** (marginal likelihood) of a model $M$ is:

$$p(y \mid M) = \int p(y \mid \theta, M)\, p(\theta \mid M)\, d\theta$$

It is the probability of the data averaged over the prior.  A model that fits the data well *and* does so with a compact prior gets a high evidence — this is the mathematical expression of **Occam's razor**: unnecessary parameters are penalized because they spread prior probability over regions that do not contribute to the fit.

### Bayes factor

Given two competing models $M_1$ and $M_2$, the **Bayes factor** is:

$$B_{12} = \frac{p(y \mid M_1)}{p(y \mid M_2)}$$

It is the factor by which the data update the prior odds in favour of $M_1$.  In log form:

$$\ln B_{12} = \ln p(y \mid M_1) - \ln p(y \mid M_2)$$

### Jeffreys scale

Interpretation of $\log_{10} B_{12}$ (evidence *in favour of* $M_1$):

| $\log_{10} B_{12}$ | Interpretation |
|---|---|
| $> 2$ | Decisive |
| $1$ – $2$ | Strong |
| $0.5$ – $1$ | Substantial |
| $< 0.5$ | Barely worth mentioning |

Negative values indicate evidence in favour of $M_2$.

### Evidence from TMCMC

TMCMC accumulates the log-evidence as a byproduct of the tempering stages.  At each stage the incremental contribution is:

$$\ln p(y \mid M) \mathrel{+}= \ln \mathbb{E}_{\theta \sim p_j}\!\left[p(y \mid \theta)^{\Delta\beta}\right]$$

which is estimated from the particles without any additional computation.  No other sampler in mcmckit computes the evidence — use TMCMC for model comparison.

### Workflow

```python
import mcmckit as mc

comp = mc.ModelComparison(
    models=[
        ("M1", problem_1, prior_samples_1),
        ("M2", problem_2, prior_samples_2),
    ],
    tmcmc_kwargs={"n_particles": 1000},
)
comp.run()
comp.summary()
```

For a standalone pairwise comparison:

```python
bf = mc.bayes_factor(result_1.log_evidence, result_2.log_evidence)
# {'log_bf': ..., 'log10_bf': ..., 'bf': ..., 'preferred': 'M1', 'evidence': 'Decisive'}
```

---

## Bayesian Model Averaging

When no model is decisively preferred — or when you want predictions that
are robust to model uncertainty — **Bayesian Model Averaging (BMA)**
combines all models weighted by their posterior probability.

### Model posterior probabilities

Assuming equal prior model probabilities $p(M_k) = 1/K$:

$$w_k = p(M_k \mid y) = \frac{p(y \mid M_k)}{\sum_j p(y \mid M_j)}$$

With informative prior model probabilities:

$$w_k \propto p(y \mid M_k)\, p(M_k)$$

### BMA posterior predictive

For any quantity of interest $Q = f(\theta)$:

$$p(Q \mid y) = \sum_{k=1}^{K} w_k\, p(Q \mid y, M_k)$$

The BMA predictive is a **mixture** of each model's posterior predictive,
weighted by $w_k$.  In practice, this is approximated by drawing
$\lfloor w_k \cdot N \rfloor$ posterior samples from model $k$ and
evaluating $f(\theta)$ at those samples.

### Behaviour

- When evidence is decisive ($w_{\text{best}} \approx 1$), BMA collapses
  to the best model's prediction.
- When two models are comparably supported ($w_1 \approx w_2 \approx 0.5$),
  BMA genuinely averages the two predictive distributions — potentially
  widening the credible band to reflect model uncertainty.
- BMA predictions are always more conservative (wider bands) than
  conditioning on a single model.

### Workflow

```python
bma = comp.predict(
    forward_models={
        "M1": lambda theta: fwd_m1(theta),
        "M2": lambda theta: fwd_m2(theta),
    },
    n_eval=1000,
)
print(bma)          # weights for each model
bma.plot_bands()    # credible band of the averaged predictive
bma.decompose()     # per-model mean, std, weight
```
