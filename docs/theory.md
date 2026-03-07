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
