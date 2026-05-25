---
aliases:
  - Learned Perceptual Image Patch Similarity
Arxiv: https://arxiv.org/abs/1801.03924
year: 2018
original title: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
date: 2026-05-16
---
---

Created with discussion with Claude Sonnet 4.6

---

A perceptual similarity loss computed in the feature space of a pretrained network. The standard reconstruction loss for image generators where pixel-space L1/L2 is inadequate.

## The Problem with Pixel-Space Losses

L1/L2 in pixel space is a poor proxy for perceptual similarity. A 1-pixel spatial shift produces high L2 error but looks identical to a human. Conversely, two images can have low L2 distance but look completely different. Minimizing L1/L2 also corresponds to maximizing a Laplacian/Gaussian likelihood, which averages over uncertainty and produces blurry outputs.

## How It Works

Pass both $x$ and $\hat{x}$ through a pretrained network (VGG or AlexNet), extract intermediate feature maps at multiple layers, compute L2 distance in that feature space, then take a weighted sum across layers:

$$
\mathcal{L}_\text{LPIPS} = \sum_l w_l \| \phi_l(x) - \phi_l(\hat{x}) \|_2^2
$$

The weights $w_l$ are learned on a dataset of human perceptual judgments (humans rating which of two distortions looks more similar to a reference). The intuition: if two images activate the same intermediate CNN features, they look similar to a human — texture, structure, and semantics are captured rather than pixel coincidence.

> [!note] LPIPS doesn't fully replace adversarial losses
> LPIPS penalizes semantic deviation well but still allows some blurring — it's computed over spatial averages of feature maps. This is why [[VQGAN]] pairs it with a PatchGAN discriminator. When some sharpness can be sacrificed for training stability, LPIPS alone is often used.
