---
date: 2026-02-08
---

*Generated via Claude 4.6 Opus, resulted from a conversation.*

---

The **score function** is the gradient of the log-probability with respect to parameters:

$$
\nabla_\theta \log p_\theta(x)
$$

It appears across statistics, machine learning, and physics — not by coincidence, but because it encodes the geometry of how probability distributions change with their parameters.

---

## Why Log?

Probabilities have multiplicative structure (independent events multiply), but calculus works in additive spaces. The logarithm bridges the two: it is the unique function (up to scale) that converts multiplicative structure to additive structure while respecting independence. This is the same reason $-\log p(k)$ appears as optimal code length in [[KL Divergence#Coding Theory Motivation|coding theory]].

For [[Cross Entropy|exponential families]] — the most natural parametric distributions — the log-probability is literally linear in the parameters:

$$
p_\theta(x) = \exp(\theta^T T(x) - A(\theta)) h(x) \quad \Longrightarrow \quad \log p_\theta(x) = \theta^T T(x) - A(\theta) + \log h(x)
$$

So the score function $\nabla_\theta \log p_\theta(x) = T(x) - \nabla A(\theta)$ is especially clean here.

---

## Two Key Properties

### Property 1: Zero Mean

The normalization constraint $\int p_\theta(x) dx = 1$ holds for all $\theta$. Differentiating both sides:

$$
\nabla_\theta \int p_\theta(x) dx = \int \nabla_\theta p_\theta(x) dx = \int p_\theta(x) \nabla_\theta \log p_\theta(x) dx = 0
$$

So $\mathbb{E}_{p_\theta}[\nabla_\theta \log p_\theta(x)] = 0$. This is not an accident — it is a geometric consequence of staying on the probability simplex. Any direction that preserves normalization must have zero expected score.

### Property 2: Tangent Vectors on the Probability Manifold

The score function components $\partial_{\theta_i} \log p_\theta(x)$ form the natural basis for the tangent space at $p_\theta$ on the [[Statistical Manifold]]. The inner product of two tangent vectors under the distribution defines the [[Fisher Information]] matrix:

$$
F_{ij}(\theta) = \mathbb{E}_{p_\theta}\left[\frac{\partial \log p_\theta}{\partial \theta_i} \cdot \frac{\partial \log p_\theta}{\partial \theta_j}\right]
$$

This gives the [[Statistical Manifold]] its Riemannian structure.

---

## The Log-Derivative Trick

The identity $\nabla_\theta p_\theta(x) = p_\theta(x) \nabla_\theta \log p_\theta(x)$ — which is just the chain rule applied to $\nabla \log p = \nabla p / p$ — converts derivatives of probabilities into expectations:

$$
\nabla_\theta \mathbb{E}_{p_\theta}[f(x)] = \nabla_\theta \int p_\theta(x) f(x) dx = \int p_\theta(x) \nabla_\theta \log p_\theta(x) f(x) dx = \mathbb{E}_{p_\theta}[f(x) \nabla_\theta \log p_\theta(x)]
$$

This matters because we can now estimate the gradient by sampling from $p_\theta$ — we never need to differentiate through the sampling process itself. This is the core of:

- **[[Policy Gradient]]** (REINFORCE): $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \nabla_\theta \log \pi_\theta(\tau)]$
- **Black-box variational inference**: gradient estimation when reparameterization is unavailable
- **Evolution strategies** and related black-box optimization methods

The zero-mean property (Property 1) is separately useful for **variance reduction**: since $\mathbb{E}[\nabla_\theta \log p_\theta] = 0$, subtracting any constant baseline $b$ from $f(x)$ does not change the expected gradient, but can dramatically reduce variance.

---

## Score w.r.t. Data: Diffusion Models

In score-based generative models, the relevant object is the score with respect to *data* rather than parameters:

$$
\nabla_x \log p_t(x)
$$

This is a vector field pointing toward higher-density regions of $p_t$. The reverse-time SDE (Anderson, 1982) uses this to denoise:

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t) d\bar{W}
$$

where $f$ is the forward drift and $g$ is the diffusion coefficient. The score function is estimated by a neural network trained via denoising score matching.

> [!note] Same formula, different spaces
> The parameter score $\nabla_\theta \log p_\theta(x)$ and the data score $\nabla_x \log p_t(x)$ are both "gradient of log-probability," but they live in completely different spaces and serve different purposes. The parameter score is a vector in parameter space that defines the [[Fisher Information]] metric and enables the [[Policy Gradient]] trick. The data score is a vector field in data space used for denoising via Langevin dynamics and reverse-time SDEs. The information geometry story (Fisher metric, natural gradient, KL curvature) does not carry over to the data score. What they share is that $\log p$ is the canonical object for doing calculus with distributions, so $\nabla \log p$ is the natural "direction" to write down — but the deep reasons each is useful are different (Tweedie's formula and reverse-time SDEs for diffusion; normalization constraint and tangent space geometry for everything else).

---

## Connections

The score function ties together several ideas:

- It defines the tangent space of the [[Statistical Manifold]]
- Its outer product gives the [[Fisher Information]] matrix (the Riemannian metric)
- Its zero-mean property enables the [[Policy Gradient]] trick
- The [[Natural Policy Gradient]] uses $F^{-1} \nabla J$ to move in the geometry the score defines
- The local quadratic approximation of [[KL Divergence]] is expressed through the Fisher metric built from score functions
