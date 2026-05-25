---
aliases:
  - VQ-GAN
Arxiv: https://arxiv.org/abs/2012.09841
year: 2020
original title: Taming Transformers for High-Resolution Image Synthesis
date: 2026-05-16
---
---

Created with conversation with Claude Sonnet 4.6

---


VQGAN trains a discrete autoencoder with a VQ codebook bottleneck (like [[VQ-VAE]]) under an adversarial + perceptual loss recipe, then fits an autoregressive transformer prior over the resulting tokens for generation. Its lasting contribution is the **VAE training recipe** — adopted by every modern latent-space generator from [[Latent Diffusion Models|LDM]] to [[Stable Diffusion 3|SD3]] to FLUX, even when the discrete bottleneck is dropped.

## The VAE Training Recipe

Four loss terms applied to a continuous or VQ-regularized autoencoder:

$$
\mathcal{L}_\text{VAE} = \mathcal{L}_\text{rec} + \lambda_\text{perc}\mathcal{L}_\text{perc} + \lambda_\text{adv}\mathcal{L}_\text{adv} + \lambda_\text{KL}\mathcal{L}_\text{KL}
$$

| Loss                                         | Purpose                                               |
| -------------------------------------------- | ----------------------------------------------------- |
| L1 reconstruction                            | Pixel-level fidelity                                  |
| [[LPIPS]] (perceptual)                       | Semantic sharpness via pretrained VGG features        |
| PatchGAN adversarial                         | Local realism; prevents blurring that LPIPS misses    |
| KL regularization (optional, tiny $\lambda$) | Keeps latent magnitude bounded; not a true bottleneck |

### L1 Reconstruction

Straightforward pixel-level L1 between input $x$ and reconstruction $\hat{x} = \mathcal{D}(\mathcal{E}(x))$. Necessary but insufficient — L1 minimization is equivalent to maximizing a Laplacian likelihood, which averages over uncertainty and produces blurry outputs wherever the decoder is unsure.

### PatchGAN — Patch-Based Adversarial Loss

**Paper:** Isola et al. 2017, "Image-to-Image Translation with Conditional Adversarial Networks" (pix2pix)

A full-image discriminator (real/fake for the whole image) is expensive, gives a single weak gradient signal, and tends to focus on global structure while ignoring local texture. PatchGAN replaces it with a fully convolutional discriminator that produces a **spatial grid of real/fake scores**, each corresponding to a local patch of the input (e.g. 70×70 pixels):

$$
\mathcal{L}_\text{adv} = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(\hat{x}))]
$$

where $D$ outputs a grid, not a scalar. The generator must fool **every patch independently** — it cannot hide blurriness in any local region. This specifically targets the high-frequency local texture that L1 and [[LPIPS]] both fail to enforce.

> [!tip] Why PatchGAN complements LPIPS
> [[LPIPS]] catches semantic/structural deviations. PatchGAN catches local sharpness failures. They cover different failure modes of pure reconstruction losses, which is why both are needed.

## Tokens + AR Transformer

The original paper pairs the discrete autoencoder with a GPT-style autoregressive transformer over the resulting tokens for generation. [[Latent Diffusion Models|LDM]] kept the VAE recipe but replaced the AR transformer with a diffusion model over continuous latents, which is the formulation that scaled.

## What "VQGAN-style" Usually Means Today

When people say "VQGAN-style training" they mean the **four-loss recipe**, not the discrete tokens or AR transformer. SD's VAE, [[DiT]]'s VAE, [[Stable Diffusion 3|SD3]]/FLUX VAEs all use this recipe with a KL-regularized continuous latent. The discrete VQ bottleneck is the part that didn't survive at scale.
