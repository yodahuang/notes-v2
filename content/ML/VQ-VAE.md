---
date: 2026-05-16
pdf: "[[vq_vae.pdf]]"
Arxiv: https://arxiv.org/abs/1711.00937
year: 2017
original title: Neural Discrete Representation Learning
---
---

This is based on a conversation with ChatGPT 5.5 followed by a conversation with Claude Sonnet 4.6 and summarized by Claude.

---

VQ-VAE replaces the continuous latent space of a [[Variational Autoencoder|VAE]] with a **discrete codebook**. The encoder output is snapped to the nearest entry in a learned embedding table instead of sampling from a Gaussian posterior.

## Motivation

### The VAE Sampling Problem

[[Variational Autoencoder|VAE]] latent spaces are hard to sample from at generation time. The KL term regularizes $q_\phi(z|x)$ toward $\mathcal{N}(0, I)$, but the aggregate posterior never exactly matches the prior. Sampling $z \sim \mathcal{N}(0,I)$ hits "holes" — regions the encoder never visited — producing blurry or incoherent outputs.

VQ-VAE sidesteps this entirely. ==The codebook _is_ the prior.== Every index is a valid code the decoder has seen during training. No holes.

### Dimensionality + Transformer-Native Generation

A $128 \times 128$ RGB image is $128 \times 128 \times 3 \times 8 = 393{,}216$ bits raw. After VQ-VAE with a $32 \times 32$ latent grid and $K = 512$ codes, the representation is $32 \times 32 \times \log_2(512) = 9{,}216$ bits — a **~42× compression**. And critically, each of those 1024 positions is a discrete index over a fixed vocabulary, so the prior model is just a language model over tokens. GPT-style autoregressive models, cross-entropy loss, joint text+image sequences — no architectural changes needed. This is how DALL-E 1 worked.

> [!tip] VQ-VAE vs VAE in one sentence
> VAE has a theoretically sampable continuous prior that's hard to sample from in practice. VQ-VAE has a discrete prior that's trivially sampable by construction, and plugs directly into transformer-based generation.

## Three Design Decisions

VQ-VAE can be understood as three independent choices stacked on each other:

**1. STE (QAT-like trick)** — the argmin nearest-neighbor lookup is non-differentiable. VQ-VAE borrows the [[Quantization-Aware Training (QAT)|QAT]] idea: forward pass uses the hard quantized value $e_k$, backward pass pretends the lookup was the identity:

$$
\frac{\partial z_q}{\partial z_e} \approx I
$$

This is the only way to get gradients through to the encoder at all.

**2. Uniform prior → drop KL** — assume $p(z) = 1/K$. Then:

$$
\text{KL}(q_\phi(z|x) | p(z)) = \log K \quad \text{(constant)}
$$

Zero gradient w.r.t. encoder parameters, so remove it from training entirely. This eliminates the KL pressure that causes posterior collapse in standard VAEs (see below).

**3. Commitment loss** — the one piece QAT doesn't need. In QAT, the quantization grid is fixed (e.g. INT8 levels) — only weights move toward it. In VQ-VAE, **both** the encoder outputs and codebook entries are learned simultaneously. Without anchoring them to each other, the encoder drifts and the codebook chases it. The two L2 terms with stop-gradients fix this by making encoder and codebook meet in the middle.

> [!note] 
> The commitment loss is not a fundamental part of the "discrete latent" idea — it's a training stability patch that arises specifically from co-training an encoder and a learned codebook together.

## Architecture

```
x → encoder → z_e(x) ∈ ℝ^D
                   ↓ nearest-neighbor lookup
              z_q(x) = e_k  (quantized)
                   ↓
             decoder → x̂
```

The codebook is $e \in \mathbb{R}^{K \times D}$: $K$ codes, each a $D$-dimensional vector. For images, the encoder produces a **spatial grid** (e.g. $32 \times 32$), and each grid cell independently picks its nearest code. One image → one $32 \times 32$ grid of indices, not a single index. $K$ and grid size are orthogonal hyperparameters — vocabulary size vs sequence length.

$$
k_{ij} = \arg\min_j | z_e(x)_{ij} - e_j |_2
$$

## Loss Function

$$
 \mathcal{L} = \underbrace{\log p(x \mid z_q(x))}_{\text{reconstruction}} + \underbrace{| \text{sg}[z_e(x)] - e |_2^2}_{\text{codebook loss}} + \underbrace{\beta | z_e(x) - \text{sg}[e] |_2^2}_{\text{commitment loss}} 
$$

|Term|Updates|Direction|Purpose|
|---|---|---|---|
|Reconstruction|encoder + decoder (via STE)|—|end-to-end reconstruction quality|
|Codebook loss|**codebook only** ($\text{sg}$ on $z_e$)|$e_k \to z_e(x)$|pull selected code toward encoder output|
|Commitment loss|**encoder only** ($\text{sg}$ on $e$)|$z_e(x) \to e_k$|prevent encoder from drifting away from codebook|

Only the selected $e_k$ receives gradient from the codebook loss — other entries are untouched for this input.

## Posterior Collapse

In a VAE, **posterior collapse** is when the encoder learns $q(z|x) \approx p(z)$ — the latent carries no information about $x$. The optimizer finds this as a local minimum when:

1. The **KL term** rewards making $q(z|x)$ uninformative (closer to prior = lower KL)
2. A **powerful decoder** can reconstruct $x$ without needing $z$ at all (e.g. autoregressive decoder conditioning on previous tokens)

Both terms are then satisfied without $z$ being useful. The KL term doesn't cause collapse by itself — it's the interaction with a decoder expressive enough to not need the latent.

VQ-VAE avoids this because the uniform prior makes the KL constant — no gradient, no pressure to make $z$ uninformative. The discrete bottleneck also forces information to flow through the codebook path. The paper demonstrates this by training a second VQ-VAE with a PixelCNN decoder on top of the first stage's latents — a setup that would cause collapse in a standard VAE.

## Two-Stage Generation

The uniform prior is a training convenience — it says "I don't care which codes get used or in what arrangement, just reconstruct well." So after training, the codebook and decoder are good, but you have no model of what _realistic_ code sequences look like. Uniform sampling produces random uncorrelated code grids → incoherent output.

The second stage fits the actual distribution of code grids from training data:

```
Stage 1: Train VQ-VAE
  learns tokenizer (encoder + codebook) + decoder

Stage 2: Train autoregressive prior over codes
  p(z_1, z_2, ..., z_T) = ∏ p(z_i | z_{<i})   [PixelCNN, Transformer]
  learns what realistic code arrangements look like

Generation:
  sample code grid from prior → decode to pixels
```

**Training the prior.** Encode all training images with the frozen encoder to get a dataset of integer grids. Then train an autoregressive model on those sequences — from the prior's perspective this is identical to language modeling: vocabulary size $K$, sequence length $H \times W$, cross-entropy loss, same sampling strategies at inference. Architecturally, modern approaches just use a transformer with 2D positional embeddings (or 2D RoPE) and flatten the code grid into a sequence — the 2D spatial structure concern that motivated PixelCNN is largely a 2017 problem, attention handles it implicitly. Sequence length is the real constraint: a $32 \times 32$ grid is 1024 tokens (fine), $64 \times 64$ is 4096 (expensive), which is part of why continuous latent diffusion (LDM) became attractive — it sidesteps autoregressive sampling cost entirely.

**Why not train jointly?** The codebook is a moving target during VQ-VAE training — codes shift meaning as the encoder learns. A jointly trained prior has to track a vocabulary that keeps changing, which is a much harder optimization. Two-stage keeps things clean: freeze the codebook first, then fit the prior to a fixed vocabulary. Joint training "is left as future research" in the paper and hasn't displaced the two-stage approach in practice.

## Relation to Other Work

| Model                            | Latent Type                 | Prior                                 | Generation                          |
| -------------------------------- | --------------------------- | ------------------------------------- | ----------------------------------- |
| [[Variational Autoencoder\|VAE]] | Continuous Gaussian         | Fixed $\mathcal{N}(0,I)$              | Sample from prior, decode           |
| VQ-VAE                           | Discrete codebook           | Uniform → learned separately          | Sample from learned prior, decode   |
| [[Latent Diffusion Model\|LDM]]  | Continuous (VQ-regularized) | Learned diffusion process             | Iterative denoising in latent space |
| DALL-E 1                         | Discrete (dVAE tokens)      | Transformer over joint token sequence | Autoregressive                      |
