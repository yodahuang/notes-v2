---
aliases:
  - VI
date: 2026-05-11
---
---

This note is from a discussion with Claude Opus 4.7 when reading [[Variational Autoencoder|VAE]] tutorial.

---

A framework for turning **inference into optimization**. The cleanest way in is through EM: VI is what EM becomes when you can't do the exact E-step.

## The problem

Given a probabilistic model $p(x, z)$ with observed $x$ and latent $z$, Bayesian inference wants the posterior:

$$
 p(z|x) = \frac{p(x|z)p(z)}{p(x)} = \frac{p(x|z)p(z)}{\int p(x|z)p(z), dz} 
$$

The denominator is intractable for most interesting models — high-dimensional integrals, mixed discrete/continuous structure, sometimes infinite-dimensional latents. Exact inference is off the table.

## Detour: EM as coordinate ascent on the ELBO

EM is usually taught as "alternate two steps until convergence," but the structural reason it works is that both steps are coordinate ascent on the [[ELBO|same objective]]. Recall the identity:

$$
 \log p_\theta(x) = \mathrm{ELBO}(q, \theta) + D_{KL}(q(z) | p_\theta(z|x)) 
$$

EM is coordinate ascent on the right-hand side:

- **E-step**: maximize over $q$ with $\theta$ fixed. If $q$ is unconstrained, the max sets $q = p_\theta(z|x)$ — the gap drops to zero, $\mathrm{ELBO} = \log p_\theta(x)$ exactly.
- **M-step**: maximize over $\theta$ with $q$ fixed. ELBO goes up (or stays).

This is why EM monotonically improves the likelihood:

- After E-step: $\mathrm{ELBO}(q_\text{new}, \theta_\text{old}) = \log p_{\theta_\text{old}}(x)$ exactly (gap closed).
- After M-step: $\mathrm{ELBO}(q_\text{new}, \theta_\text{new}) \geq \mathrm{ELBO}(q_\text{new}, \theta_\text{old})$.
- Always: $\log p_{\theta_\text{new}}(x) \geq \mathrm{ELBO}(q_\text{new}, \theta_\text{new})$.

Chained: $\log p_{\theta_\text{new}}(x) \geq \log p_{\theta_\text{old}}(x)$ — the likelihood never decreases. The ELBO is a lower bound that _touches_ the true log-likelihood at every E-step, so M-step improvements transfer directly to the real objective. That's the structural reason EM works, beyond "alternate A then B."

k-means is a degenerate case: EM on an isotropic GMM in the limit $\sigma \to 0$. The posterior over cluster assignments collapses to a point mass on the nearest centroid (hard instead of soft assignment), and the M-step becomes "centroid = mean of assigned points." All of k-means's pathologies (local minima, init sensitivity) are inherited from EM, with extra rigidity from hard assignments.

## VI as EM with a restricted $q$

The exact E-step requires $p_\theta(z|x)$ — but that's the intractable thing. VI's move: pick a tractable family $\mathcal{Q}$ and settle for the best $q$ within it:

$$
 q^*(z) = \arg\min_{q \in \mathcal{Q}} D_{KL}(q(z) | p_\theta(z|x)) = \arg\max_{q \in \mathcal{Q}} \mathrm{ELBO}(q, \theta) 
$$

The gap doesn't fully close — there's a residual $D_{KL}(q^* | p_\theta(z|x)) > 0$ — so the ELBO stays a _strict_ lower bound. M-step improvements still raise the bound, but the likelihood guarantee weakens: you're now optimizing a surrogate that may not perfectly track $\log p_\theta(x)$. The looser $\mathcal{Q}$, the closer to EM; the tighter, the more tractable.

Every variant of VI is a different choice of $\mathcal{Q}$ and a different way to do the maximization.

## Variants

**Mean-field VI.** Factorize $q(z) = \prod_i q_i(z_i)$ — assume the latents are independent under $q$. Coordinate-ascent updates have closed forms for conjugate models. Cheap, but the factorization ignores posterior correlations between latents, which tends to underestimate posterior variance.

**Structured VI.** Allow some dependence in $q$ but not full — tree-structured, chain-structured. More expressive than mean-field, more expensive.

**[[Amortized variational inference]].** Instead of separately fitting $q^{(i)}(z)$ for every datapoint, share a neural network $\phi$ that maps $x \mapsto q_\phi(z|x)$. The [[Variational Autoencoder|VAE]] encoder is the canonical example.

**Stochastic VI (SVI).** Minibatch gradients on the ELBO instead of full-batch coordinate ascent. The thing that made VI work on web-scale topic models.

**Black-box VI (BBVI).** Generic gradient estimators ([[Score function]] or [[Reparameterization trick]]) so you don't need conjugacy. Combined with neural-network $q$, this is what powers modern deep latent variable models.

## Examples beyond VAE

- **LDA / topic models**: mean-field VI over per-document topic mixtures and per-word topic assignments. The application that put VI on the map.
- **Bayesian neural nets**: $q$ over weights (mean-field Gaussian, MC dropout, normalizing flow).
- **State-space models**: VI as an alternative to [[Extended Kalman Filter|EKF]]/particle filters for nonlinear/non-Gaussian dynamics.
- **Stochastic block models, mixed-membership models, HDPs**: pre-deep-learning Bayesian nonparametrics relied heavily on VI.
- **DDPM / diffusion**: the training objective is a per-step [[ELBO]] on a $T$-latent hierarchical model. The equivalent view via [[Score matching]] regresses the score field directly.