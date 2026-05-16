---
date: 2026-05-11
---
---

This note is from a discussion with Claude Opus 4.7 when reading [[Variational Autoencoder|VAE]] tutorial.

---

A variant of [[Variational inference]] where, instead of separately optimizing $q(z)$ for each datapoint $x^{(i)}$, a shared neural network maps $x \mapsto q_\phi(z|x)$. The same parameters $\phi$ serve all datapoints.

## Why amortize

In **classical (pre-amortized) VI**, the variational distribution has free parameters fit per datapoint. With a diagonal-Gaussian family, each $x^{(i)}$ has its own $(\mu^{(i)}, \sigma^{(i)})$, fit by gradient ascent (or coordinate ascent for conjugate models) on the per-datapoint ELBO:

```
for each x_i in dataset:
    fit (μ_i, σ_i) by gradient ascent on ELBO_i
```

That's $N$ separate optimization problems. At inference time on a new $x$, you run optimization again. See [[Variational inference]] for the broader framework this slots into.

Amortized VI replaces this with one network:

```
train neural net φ: x -> q_φ(z|x) once
at inference: forward pass gets q_φ(z|x_new) in O(1)
```

The cost of inference is amortized across the training set, hence the name.

## The amortization gap

The downside: $q_\phi(z|x)$ for a fixed network is generally worse than the per-datapoint optimum $q^{(i)*}(z)$. Cremer et al. (2018) decompose the total looseness of the bound:

$$
 \underbrace{\log p(x) - \mathrm{ELBO}_{q_\phi}}_{\text{total gap}} = \underbrace{\log p(x) - \mathrm{ELBO}_{q^*_\text{family}}}_{\text{approximation gap}} + \underbrace{\mathrm{ELBO}_{q^*_\text{family}} - \mathrm{ELBO}_{q_\phi}}_{\text{amortization gap}} 
$$

- **Approximation gap**: the variational family $\mathcal{Q}$ (e.g. diagonal Gaussian) can't represent the true posterior.
- **Amortization gap**: even within $\mathcal{Q}$, the network doesn't reach the per-datapoint optimum.

Amortization is the price of fast inference, paid as a slacker [[ELBO]].

## VAE as the canonical case

In a [[Variational Autoencoder|VAE]], the encoder $f_\phi: x \mapsto (\mu_\phi(x), \sigma_\phi(x))$ is the amortizer. The variational family is diagonal Gaussian:

$$
 q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \mathrm{diag}(\sigma_\phi(x)^2)) 
$$

Trade-offs:

- **Scalability**: $O(1)$ amortized inference is what makes deep latent variable models trainable on millions of samples.
- **Generalization**: the encoder must handle unseen $x$ at test time. Classical per-datapoint VI doesn't address this at all — every new datapoint is a fresh optimization.
- **Posterior collapse**: when the decoder is powerful enough to model $p(x)$ without using $z$, the encoder collapses to the prior — $q_\phi(z|x) \approx p(z)$ — because the KL term pulls $q$ to $p(z)$ and the reconstruction term doesn't penalize it. Common with autoregressive decoders. One of the motivating problems for [[VQ-VAE]] (discrete codes force the decoder to use them).

## Closing the amortization gap

- **Semi-amortized VI** (Kim et al. 2018): run a few gradient steps starting from $q_\phi(z|x)$ to refine per-datapoint at training/inference time.
- **Iterative amortized inference** (Marino et al. 2018): replace the encoder with a learned optimizer that iteratively improves $q$, generalizing the "encode in one shot" view.
- **Richer $q$ families**: stack a normalizing flow on top of the encoder output to expand $\mathcal{Q}$ — this reduces the approximation gap, sometimes at the cost of amortization gap.