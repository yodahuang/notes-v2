---
aliases:
  - Pathwise gradient
date: 2026-05-11
---
---

This note is from a discussion with Claude Opus 4.7 when reading [[Variational Autoencoder|VAE]] tutorial.

---

The trick that makes [[Variational Autoencoder|VAE]] training work end-to-end. It lets gradients flow through a sampling step.

## The problem

We want to maximize an objective of the form:

$$ \mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z)}[f(z)] $$

Gradient descent needs $\nabla_\phi \mathcal{L}$. But $\phi$ controls the _distribution we sample from_, not the function inside, so:

$$ \nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] \neq \mathbb{E}_{q_\phi(z)}[\nabla_\phi f(z)] $$

The expectation itself depends on $\phi$ through the sampling density. The gradient doesn't commute with the expectation.

## The trick

If $z$ can be written as a deterministic function of $\phi$ and a $\phi$-free noise variable:

$$ z = g_\phi(\epsilon), \quad \epsilon \sim p(\epsilon) $$

then the expectation rewrites with the _fixed_ base distribution outside:

$$ \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{p(\epsilon)}[f(g_\phi(\epsilon))] $$

Now the gradient passes through cleanly:

$$ \nabla_\phi \mathbb{E}_{p(\epsilon)}[f(g_\phi(\epsilon))] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))] $$

A Monte Carlo estimate: sample $\epsilon$, compute $\nabla_\phi f(g_\phi(\epsilon))$ via autograd, done.

## Gaussian case (the VAE one)

For $q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi(x)^2 I)$:

$$ z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$

The encoder outputs $\mu$ and $\log \sigma^2$ (logvar, for numerical stability — exponentiating keeps $\sigma^2$ positive without constraints). The sample $z$ is a deterministic function of $\mu$, $\sigma$, $\epsilon$; backprop flows through $\mu$ and $\sigma$ into the encoder.

This is why VAE encoder heads output two things rather than a sample.

## Versus the [[Score function]] estimator

The score-function ([[Policy Gradient|REINFORCE]]) estimator works for any $q_\phi$ without requiring a reparameterization:

$$ \nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z) \nabla_\phi \log q_\phi(z)] $$

It's universal but high-variance — $f(z)$ multiplies the score, so noise in $f$ amplifies into the gradient estimate. Variance reduction (baselines, control variates) helps but rarely closes the gap.

Reparameterization is lower-variance because it uses gradient information from $f$ directly (pathwise derivative carries shape information about $f$, not just scalar values). When applicable, prefer it.

## Other applications

The same pathwise-gradient construction appears wherever a network needs to inject learnable-scale noise and still get gradients through the noise scale:

- **Noisy top-k gating in [[Mixture of Experts]]** (Shazeer 2017): $H(x)_i = (xW_g)_i + \epsilon \cdot \mathrm{softplus}((xW_n)_i)$ with $\epsilon \sim \mathcal{N}(0,1)$. Structurally identical to $z = \mu + \sigma \odot \epsilon$ but for gating logits rather than posterior samples. Purpose is exploration and load balancing across experts, not posterior approximation, but the mechanism is the same pathwise gradient through $W_n$.
- **Stochastic policies in continuous-control RL** (SAC): action $a = \mu_\phi(s) + \sigma_\phi(s) \odot \epsilon$, gradients flow into the policy network through the action. Replaces high-variance [[Score function]] policy gradients with low-variance pathwise gradients of the Q-value — one of the reasons SAC works well at scale.

## Limitations

Reparameterization requires $z$ to be a differentiable function of $\phi$ given $\epsilon$. This rules out:

- **Discrete latents**: can't differentiate through a categorical sample. Workarounds: Gumbel-Softmax (continuous relaxation, biased but low-variance), straight-through estimator (zero-bias for the forward pass, biased gradient — the choice in [[VQ-VAE]]), or fall back to [[Score function]] gradients. See also [[DPO]] for discrete sampling.
- **Distributions without nice reparameterizations**: gamma, Dirichlet, etc. Generalized reparameterization (Ruiz et al. 2016) and implicit reparameterization (Figurnov et al. 2018) extend the trick using inverse CDFs or rejection sampling.

The discrete case is exactly what motivates VQ-VAE's codebook + straight-through gradient design.