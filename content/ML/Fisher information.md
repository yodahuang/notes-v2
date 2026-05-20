---
date: 2026-02-08
---

*Generated via Claude 4.6 Opus, resulted from a conversation.*

---

The **Fisher information matrix** measures how sensitively a probability distribution $p_\theta$ responds to changes in its parameters. It is defined as the covariance of the [[Score Function]]:

$$
F(\theta) = \mathbb{E}_{p_\theta}\left[\nabla_\theta \log p_\theta(x) \nabla_\theta \log p_\theta(x)^T\right]
$$

Since $\mathbb{E}[\nabla_\theta \log p_\theta] = 0$ (the score has zero mean), this is both the second moment and the covariance.

---

## Equivalent Form

Under regularity conditions (exchange of differentiation and integration), there is an equivalent expression as the negative expected Hessian of the log-likelihood:

$$
F(\theta) = -\mathbb{E}_{p_\theta}\left[\nabla_\theta^2 \log p_\theta(x)\right]
$$

The equivalence follows from differentiating the zero-mean identity $\int p_\theta \nabla_\theta \log p_\theta dx = 0$ a second time. This form is often more convenient for computation, especially in exponential families where $\nabla^2 \log p_\theta$ has a clean closed form.

---

## As a Riemannian Metric

The Fisher information matrix is the unique (up to scale) Riemannian metric on the [[Statistical Manifold]] that is invariant under sufficient statistics — this is Čencov's theorem.

Concretely, it defines the infinitesimal distance between nearby distributions:

$$
ds^2 = \sum_{i,j} F_{ij}(\theta) d\theta_i d\theta_j
$$

This tells you that some parameter directions change the distribution a lot (large eigenvalues of $F$) while others barely affect it (small eigenvalues). Euclidean distance in parameter space ignores this entirely.

The connection to [[KL Divergence]] is direct: for nearby distributions $p_\theta$ and $p_{\theta + d\theta}$, the Taylor expansion of KL divergence gives

$$
D_{KL}(p_\theta \| p_{\theta + d\theta}) \approx \frac{1}{2} d\theta^T F(\theta) d\theta
$$

The first two terms of the expansion vanish (the zeroth by $D_{KL}(p \| p) = 0$, the first because $\nabla_{\theta'} D_{KL}(p_\theta \| p_{\theta'})\big|_{\theta'=\theta} = 0$). So the Fisher matrix is the Hessian of KL divergence at coincidence — it *is* the local curvature of KL divergence.

---

## Natural Gradient

Standard gradient descent treats all parameter directions equally — it uses the Euclidean metric $I$ (the identity matrix). But on the [[Statistical Manifold]], the natural metric is the Fisher matrix. The **natural gradient** corrects for this:

$$
\tilde{\nabla}_\theta J = F(\theta)^{-1} \nabla_\theta J(\theta)
$$

This is the steepest ascent direction in the geometry of distributions rather than in parameter space. It is invariant to reparameterization: if you change coordinates $\theta \to \phi(\theta)$, the natural gradient update produces the same change in $p_\theta$.

This is exactly what the [[Natural Policy Gradient]] uses, and it motivates [[TRPO]], which enforces a trust region directly in KL divergence.

---

## Cramér-Rao Bound

For any unbiased estimator $\hat{\theta}$ of $\theta$, the covariance of the estimator is bounded below:

$$
\text{Cov}(\hat{\theta}) \succeq F(\theta)^{-1}
$$

in the positive semidefinite sense. This means the Fisher information quantifies the **best possible precision** of any unbiased estimator. High Fisher information at $\theta$ means the data is informative about $\theta$ — the distribution changes rapidly, so observations can pin down the parameter.

---

## Connection to the Information Filter

The "information" in the [[Kalman Filter]]'s information form is genuinely Fisher information. For a Gaussian $\mathcal{N}(\mu, \Sigma)$ with known covariance, the Fisher information about the mean from a single observation is $\Sigma^{-1}$ — the precision matrix.

In the information filter's [[Gaussian#Natural parameterization|natural parameterization]] ($P = \Sigma^{-1}$, $J = \Sigma^{-1}\mu$), the measurement update becomes:

$$
P_{\text{new}} = P_{\text{old}} + C^T Q^{-1} C
$$

The term $C^T Q^{-1} C$ is exactly the Fisher information that observation $z_t$ provides about the state $x_t$. Conditioning on a new measurement = adding its Fisher information to the current precision. This is why the natural parameterization makes the update additive — Fisher information from independent observations adds.

The Kalman filter's covariance $\Sigma_t$ achieves the Cramér-Rao bound: it is the minimum-variance estimator for the linear-Gaussian setting, and $\Sigma_t^{-1}$ is the total accumulated Fisher information from all observations.

---

## Examples

**Gaussian $\mathcal{N}(\mu, \sigma^2)$** with parameters $\theta = (\mu, \sigma^2)$:

$$
F = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}
$$

The off-diagonal is zero because mean and variance are informationally orthogonal. Estimating the mean is easier (scales as $1/\sigma^2$) than estimating the variance (scales as $1/\sigma^4$). The natural gradient would take larger steps in the variance direction to compensate.

**Categorical distribution** with probabilities $(p_1, \ldots, p_K)$: the Fisher metric is $F_{ij} = \delta_{ij}/p_i$, which is the metric that makes the [[Statistical Manifold]] of categorical distributions a sphere under the square-root parameterization $\sqrt{p_i}$.
