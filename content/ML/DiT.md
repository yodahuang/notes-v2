---
date: 2026-05-17
year: 2022
Arxiv: https://arxiv.org/abs/2212.09748
original title: Scalable Diffusion Models with Transformers
pdf: "[[dit.pdf]]"
---
---

The following notes comes from my brief conversation with Claude Sonnet 4.6. 

---

## Core Idea

[[LDM]] (Latent Diffusion Models) with the UNet backbone swapped for a Vision Transformer. Patchify the VAE latent, run it through a ViT, predict noise. The architecture change is straightforward — the contribution is in the conditioning mechanism and the scaling analysis.

![[dit_architecture.png]]

## Conditioning: AdaLN-Zero

### Standard LayerNorm

LayerNorm normalizes activations then applies static learned affine parameters:

$$
\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

$\gamma$ and $\beta$ are fixed after training — the same for every sample and every timestep.

### AdaLN

Adaptive LayerNorm makes $\gamma$ and $\beta$ ==dynamic== — predicted from the conditioning signal $c$ (timestep embedding + class embedding):

$$
\gamma(c),\ \beta(c) = \text{MLP}(c)
$$

$$
\text{AdaLN}(x, c) = \gamma(c) \cdot \frac{x - \mu}{\sigma} + \beta(c)
$$

The network learns timestep-dependent scaling: "at $t=500$, modulate activations this way; at $t=50$, modulate differently." Same idea as StyleGAN's AdaIN — the style/conditioning vector controls the affine transform of every norm layer.

### AdaLN-Zero

DiT adds a third MLP output $\alpha$ that gates the residual connection:

$$
\gamma,\ \beta,\ \alpha = \text{MLP}(c)
$$

$$
x \leftarrow x + \alpha \odot \text{Block}(\text{AdaLN}(x,\ \gamma,\ \beta))
$$

The **Zero** part: the final linear projection in the MLP is ==zero-initialized==. At init, $\alpha = 0$, so every block is a pure identity map. Blocks "turn on" gradually during training as the MLP learns nonzero outputs.

> [!note] Why this helps
> Same intuition as ReZero / FixUp — zero-init residuals stabilize early training when conditioning signals are random. The model starts as a passthrough and learns to activate each block.

### Why AdaLN beats the alternatives

DiT ablates four conditioning strategies:

| Strategy        | Mechanism                                | Issue                                             |
| --------------- | ---------------------------------------- | ------------------------------------------------- |
| In-context      | Append timestep/class tokens to sequence | Wastes sequence capacity, no per-layer modulation |
| Cross-attention | Attend to conditioning tokens            | Overkill for low-dim signal (just $t$ + class)    |
| AdaLN           | Dynamic $\gamma, \beta$ per layer        | ✓                                                 |
| **AdaLN-Zero**  | AdaLN + $\alpha$ gating + zero-init      | ✓✓ best FID                                       |

AdaLN is cheap (one small MLP per layer), injects conditioning everywhere, and adds no sequence length overhead.

## Actual Contribution

The architecture itself is not novel — anyone familiar with ViT and LDM could sketch it. The real contributions are:

**1. Scaling law for diffusion.** DiT follows a clean compute scaling law: ==Gflops (not parameter count) predict FID==. Bigger model + more compute → monotonically better, predictably. The [[Scaling Law|Chinchilla]] story applied to image diffusion.

**2. Conditioning ablation.** The comparison across in-context / cross-attention / adaLN / adaLN-Zero is the empirical work that earns the design choice.

**3. Clean baseline.** DiT-XL/2 was SoTA on class-conditional ImageNet 256×256 at the time, with a simple, reproducible setup. Essentially every serious diffusion model since — SD3, Flux, Sora (presumed) — is DiT-derived.

> [!tip] Genre
> Same genre as [[ConvNeXT]] "obvious idea, done carefully at scale with the right ablation." Not flashy science, but extremely influential because it provides a clean foundation the community can build on.