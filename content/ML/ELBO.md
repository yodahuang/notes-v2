---
aliases:
  - Evidence Lower Bound
date: 2026-05-11
---
---

This note is from a discussion with Claude Opus 4.7 when reading [[Variational Autoencoder|VAE]] tutorial.

---

The universal objective of [[Variational inference]]. Given a latent variable model $p_\theta(x, z) = p_\theta(x|z)p(z)$ with intractable posterior $p_\theta(z|x)$, the ELBO is a tractable lower bound on $\log p_\theta(x)$ that we can actually optimize.

## Definition

$$ \mathrm{ELBO}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) | p(z)) $$

Equivalent rewriting:

$$ \mathrm{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x, z) - \log q_\phi(z|x)] $$

(Use $\log p(x,z) = \log p(x|z) + \log p(z)$ and pull the $\log p(z) - \log q$ terms together.)

## The identity

Start from $D_{KL}(q_\phi | p_\theta(z|x))$ and expand using Bayes' rule $\log p_\theta(z|x) = \log p_\theta(x,z) - \log p_\theta(x)$:

$$ \begin{aligned} D_{KL}(q_\phi(z|x) | p_\theta(z|x)) &= \mathbb{E}_{q_\phi}[\log q_\phi(z|x) - \log p_\theta(z|x)] \\ &= \mathbb{E}_{q_\phi}[\log q_\phi(z|x) - \log p_\theta(x,z)] + \log p_\theta(x) \end{aligned} $$

Rearranging gives the central identity:

$$ \log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(x,z) - \log q_\phi(z|x)]}_{\mathrm{ELBO}} + D_{KL}(q_\phi(z|x) | p_\theta(z|x)) $$

Three facts fall out:

1. Since KL is nonnegative, $\log p_\theta(x) \geq \mathrm{ELBO}$ — hence "lower bound."
2. The gap **is** the KL between the approximate and true posterior.
3. The bound is tight when $q_\phi(z|x) = p_\theta(z|x)$.

This is the form to keep in your head — "evidence = ELBO + gap" decomposes the intractable left side into one computable piece (ELBO) and one positive piece (the gap) that vanishes when $q$ matches the true posterior.

### Alternative derivation: Jensen

A shorter route that gives the bound but doesn't expose what the gap _is_:

$$ \log p_\theta(x) = \log \int p_\theta(x, z), dz = \log \mathbb{E}_{q_\phi(z|x)}!\left[\frac{p_\theta(x,z)}{q_\phi(z|x)}\right] $$

Apply Jensen ($\log$ is concave, so $\log \mathbb{E}[\cdot] \geq \mathbb{E}[\log \cdot]$):

$$ \log p_\theta(x) \geq \mathbb{E}_{q_\phi}!\left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right] = \mathrm{ELBO} $$

Clean, but no gap term — for the gap, use the KL-identity derivation above.

## The gap shrinks for free

The puzzling part: we want to minimize the gap $D_{KL}(q_\phi | p_\theta(z|x))$, but $p_\theta(z|x)$ is exactly the thing we can't compute. How does optimization actually close it?

Look at gradients with respect to the encoder parameters $\phi$ only (decoder $\theta$ held fixed). Since $\log p_\theta(x)$ has no $\phi$ dependence:

$$ \frac{\partial}{\partial \phi}\log p_\theta(x) = 0 $$

From the identity:

$$ \frac{\partial \mathrm{ELBO}}{\partial \phi} = -\frac{\partial D_{KL}(q_\phi | p_\theta(z|x))}{\partial \phi} $$

**Every gradient step on $\phi$ that increases the ELBO is exactly a gradient step that decreases the KL gap.** You never compute the gap, but encoder optimization closes it anyway. That's the "for free" part.

For the decoder parameters $\theta$, both terms on the right change when you update — increasing the ELBO over $\theta$ pushes $\log p_\theta(x)$ up _and_ may shift the gap either way. But in joint optimization, the encoder keeps closing the gap while the decoder improves the fit. This is the same identity that makes EM work; an exact E-step ($q \leftarrow p_\theta(z|x)$) sets the gap to zero and the ELBO becomes the true log-likelihood. See [[Variational inference]] for the EM-as-VI framing in full.

## Why reverse KL

The chosen direction $D_{KL}(q | p(z|x))$ rather than $D_{KL}(p(z|x) | q)$ is forced by tractability — see [[KL divergence#Forward vs. Reverse KL Geometric Intuition]] for the mode-seeking vs mass-covering distinction. The short version: expectations under $q$ are computable because we sample from $q$; expectations under $p(z|x)$ are not because $p(z|x)$ is the intractable object. The forward KL has the right "covering" behavior in principle but is unusable here.

## Where it shows up

- **[[Variational Autoencoder (VAE)]]**: $q_\phi$ is a neural encoder, $p_\theta$ is a neural decoder, the ELBO is the training loss.
- **[[Amortized variational inference]]**: the general framework VAE instantiates.
- **DDPM / diffusion**: the noising chain $x_0 \to x_1 \to \cdots \to x_T$ is a hierarchical latent variable model with frozen forward $q$ and learned reverse $p_\theta$. The training objective is a per-step ELBO that reparameterizes to noise-prediction MSE (Ho et al. 2020). The score-regression view in [[Score matching]] is equivalent up to weighting.
- **Bayesian neural nets**: $q_\phi$ is a distribution over weights (mean-field Gaussian, MC dropout, normalizing flow) trained against a weight prior.
- **LDA / topic models**: VI's pre-deep-learning killer app. $q$ factors over per-document topic mixtures and per-word topic assignments.