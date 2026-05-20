---
date: 2026-05-19
pdf: "[[stable_diffusion_3.pdf]]"
year: 2024
Arxiv: https://arxiv.org/abs/2403.03206
original title: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
---
---

The following notes come from my discussion with Claude Sonnet 4.6

---

# Stable Diffusion 3 — Scaling Rectified Flow Transformers

> [!abstract] 
> SD3 introduces the **Multimodal Diffusion Transformer (MMDiT)** — a rectified-flow model that treats text and image tokens as two parallel streams with separate weights but joint attention. Key contributions: (1) a careful empirical study of timestep sampling schedules, (2) the MMDiT architecture, (3) improved text conditioning via multiple encoders, and (4) resolution-aware positional encoding and timestep shifting.


---

## 1. Timestep Sampling: Why Log-Normal?

### 1.1 The Core Problem

Rectified flow trains with a uniform distribution over $t \in [0,1]$, but the prediction difficulty is _not_ uniform across timesteps. At the extremes:

- $t \approx 0$: image is nearly clean → optimal prediction is just the mean of $p_0$ (trivial)
- $t \approx 1$: image is nearly pure noise → optimal prediction is the mean of $p_1$ (also trivial)

The hard, information-rich region is the **middle**, where signal and noise are genuinely mixed. We want to oversample it.

### 1.2 Log-SNR: The Natural Coordinate

For rectified flow with $\alpha_t = t$, $\beta_t = 1-t$, the log signal-to-noise ratio is:

$$
\lambda(t) = \log\frac{\alpha_t^2}{\beta_t^2} = \log\frac{t^2}{(1-t)^2} = 2\log\frac{t}{1-t} = 2\text{logit}(t)
$$

$\lambda$ is the natural axis for measuring difficulty: each unit of $\lambda$ corresponds to a doubling/halving of the signal-to-noise ratio. At $t=0.5$, $\lambda=0$ — equal signal and noise. As $t\to 0$, $\lambda\to-\infty$ (all signal); as $t\to 1$, $\lambda\to+\infty$ (all noise).

### 1.3 Why Log-Normal Specifically

If prediction difficulty is roughly uniform per unit of $\lambda$ — each octave of SNR deserves equal training — then we want $\lambda$ uniformly distributed. Since $\lambda = 2,\text{logit}(t)$, pulling $\lambda \sim \mathcal{U}$ back through the change of variables gives $t$ distributed as **logit-normal**: place a Gaussian on $\text{logit}(t)$, then invert.

Concretely, the density is:

$$
\pi_\text{ln}(t;, m, s) = \frac{1}{s\sqrt{2\pi}}\cdot\frac{1}{t(1-t)}\exp\left(-\frac{(\text{logit}(t)-m)^2}{2s^2}\right)
$$

- **Location $m$**: shifts weight toward data ($m<0$) or noise ($m>0$). Default $m=0$ peaks at $t=0.5$.
- **Scale $s$**: controls width. Larger $s$ → flatter, closer to uniform.
- **Tails vanish at 0 and 1**: the $t(1-t)$ denominator is the Jacobian of the logit transform — no wasted signal at the trivial endpoints.

### 1.4 Comparison: π Series Prioritizes Opposite End

SD3's log-normal peaks in the **middle** because images need both coarse structure _and_ fine detail — no strong asymmetry.

[[Pi 0]] and the RECAP series (robot learning) do the opposite: **up-weight high-noise timesteps** (large $t$, low SNR). For robot action generation, coarse trajectory correctness dominates — a wrong global motion plan fails the task regardless of fine detail. Fine denoising at small $t$ is cheap to recover from; coarse denoising at large $t$ is not. Opposite asymmetry from image generation.

> [!tip] Key takeaway
> The right timestep distribution reflects the **loss asymmetry of your task**. SD3: symmetric → log-normal. [[Pi 0]]: coarse matters more → up-weight high noise.

---

## 2. Architecture: MMDiT

### 2.1 Overview
![[mmdit.png]]

The architecture has three components: text conditioning, the MMDiT backbone, and the VAE. See [[DiT]] and [[Flow Matching]] for the underlying building blocks.

### 2.2 Two Conditioning Signals: `c_vec` → `y` and `c_ctxt`

The text is encoded by **three frozen models** and split into two representations with fundamentally different roles:

**`c_vec` (pooled, global) → becomes `y`**

[[CLIP]]-L and OpenCLIP-G pooled outputs are concatenated → $\mathbf{c}_\text{vec} \in \mathbb{R}^{2048}$. Combined with the timestep embedding, fed through an MLP to produce scale/shift/gate parameters of **adaLN-zero** at every MMDiT block.

- Carries: _what kind of image is this?_ — holistic semantic gist
- The timestep $t$ naturally lives here too: also a scalar global signal
- Mechanism: modulates the _gain and bias_ of every activation uniformly

**`c_ctxt` (full sequence, local) → joint attention**

CLIP penultimate hidden states (zero-padded 2048→4096) concatenated with T5-XXL hidden states → $\mathbf{c}_\text{ctxt} \in \mathbb{R}^{154 \times 4096}$ (77 CLIP + 77 T5 tokens). Concatenated with image patch tokens for **bidirectional joint self-attention**.

- Carries: _which words say what, and where?_ — token-level spatial grounding
- Mechanism: cross-token attention lets patches route to relevant words

The 77-token limit on CLIP comes from its fixed positional embedding table (76 content positions + `[EOS]`). T5 is also truncated to 77 for uniform concatenation.

> [!question] Do we actually need `y` / `c_vec` in addition to `c_ctxt`? 
> 
> Honestly unclear. The pooled vector has a different representational character — CLIP's `[EOS]` token is contrastively trained for global image-text similarity, not token-level semantics. adaLN is also a cheaper and more direct broadcast path than attention.
> 
> But there is no ablation isolating `c_vec`'s contribution while keeping `c_ctxt`. It was inherited from SDXL (found empirically to help) and never seriously questioned. The timestep needs to live _somewhere_ global — adaLN is the natural home — and `c_vec` is just concatenated to it cheaply.

### 2.3 MMDiT as Hard-Coded 2-Expert MoE

The dual-stream design is cleanly understood as a **hard-routed Mixture of Experts by modality**:

- Text tokens → text expert weights (Q, K, V, FFN)
- Image tokens → image expert weights (separate Q, K, V, FFN)
- Routing: 100% hard, determined by token type — no learned router
- Cross-expert interaction: shared attention score matrix — both streams' keys and queries concatenated before softmax

Structurally identical to [[Pi 0]]'s action/observation expert split: VLM backbone handles observation tokens, action expert handles action tokens, interact only through attention. The two modalities have sufficiently different statistical distributions that separate weight matrices are worth the cost — but joint attention is still needed for cross-modal grounding.

### 2.4 Improved Text Encoders and Synthetic Captions

SD3 uses larger encoders than previous SD versions ([[CLIP]] bigG + T5-XXL vs CLIP-L alone). Training images are re-captioned using a separate VLM to generate **dense, descriptive context labels** — the model sees both the original human caption and the synthetic VLM caption at a **50/50 ratio**. See [[Hi Robot]] for a similar synthetic captioning approach applied to robot data. The synthetic captions are more compositionally detailed and help the model learn fine-grained attribute binding.

---

## 3. QK Normalization

SD3 applies **RMSNorm with learnable scale** to Q and K vectors in both streams before computing attention logits.

**The problem:** when fine-tuning at higher resolutions, patch token count grows quadratically, attention logits grow unboundedly → entropy explodes → training diverges. First documented for large ViTs (Dehghani et al. 2023, ViT-22B).

**Why it works:** normalizing Q and K bounds all attention logits by $|q||k| \leq C$, preventing softmax saturation and "winner-take-all" collapse.

**Is it specific to diffusion models?** No — but more critical for them. Variable resolution training creates extreme sequence length variation, triggering logit explosion more acutely than fixed-length LLM training. LLMs that use it: Gemma 2/3, OLMo 2, OpenELM. One notable incompatibility: QK-norm requires materializing full Q/K vectors, making it incompatible with [[MLA]] (DeepSeek's multi-latent attention), where Q/K are reconstructed from low-rank factors at inference time.

Adopted as a headline change in **SD3.5** (enabling stable 8B training), and present in **Flux.1** (both doubleand single-stream blocks).

> [!tip] 
> QK-norm constrains attention scores to a hypersphere — all comparisons become cosine similarities, bounded in $[-1,1]$ regardless of depth or sequence length.

---

## 4. Positional Encoding for Varying Aspect Ratios

SD3 uses 2D sinusoidal frequency embeddings over a canonical coordinate grid. The challenge: embeddings must be **physically consistent** across aspect ratios — "far right" should mean the same thing in square, wide, or tall images.

**The approach:** build a canonical grid spanning the maximum extent across all aspect ratio buckets:

$$
\text{grid}_h = \left(\frac{p - \frac{h_\text{max} - s}{2}}{S/256}\right)_{p=0}^{h_\text{max}-1}, \quad \text{similarly for width}
$$

where $s = S/16$ is the latent size (after [[Variational Autoencoder|VAE]] + patching), $h_\text{max}$ is the tallest latent across all buckets, $S$ is target resolution. For any specific image, take a **center crop** of this canonical grid.

**Why center-crop rather than interpolate?** ViT-style interpolation distorts physical meaning — "position 50" means a different spatial fraction at different aspect ratios. Center-cropping from a fixed coordinate system preserves it: every position value corresponds to a fixed spatial distance regardless of image shape.

> [!note] RoPE supersedes this 
> Flux.1 replaces sinusoidal absolute embeddings with RoPE — positions encoded as rotations of Q/K vectors, so only _relative_ positions enter the attention score. The canonical-grid + center-crop mechanism becomes unnecessary. RoPE generalizes to unseen resolutions without any coordinate bookkeeping.

---

## 5. Resolution-Dependent Timestep Shifting

**Core observation:** the same $t$ destroys different amounts of signal at different resolutions. A model trained at $256^2$ is miscalibrated at $1024^2$.

**Derivation via uncertainty matching:**

Consider a constant image (every pixel = $c$) at resolution $n = H \times W$. Forward process: $z_t = (1-t)c\mathbf{1} + t\epsilon$. To recover $c$, average the pixels:

$$
\hat{c} = \frac{1}{1-t}\cdot\frac{1}{n}\sum_i z_{t,i}, \qquad \sigma(\hat{c}) = \frac{t}{(1-t)\sqrt{n}}
$$

Higher resolution → more pixels → lower uncertainty at the same $t$. Matching uncertainty $\sigma(t_n, n) = \sigma(t_m, m)$:

$$
t_m = \frac{\alpha, t_n}{1 + (\alpha-1),t_n}, \qquad \alpha = \sqrt{\frac{m}{n}}
$$

For $m > n$: $\alpha > 1$, so $t_m > t_n$ — **shift toward later timesteps** (more noise) at higher resolution. In log-SNR coordinates, this is simply a constant translation: $\lambda_{t_m} = \lambda_{t_n} - 2\log\alpha$.

In practice, $\alpha = 3.0$ for $1024^2$ training (found via human preference study), applied at both training and sampling time.

> [!note] 
> The derivation assumes a constant image, which is unrealistic. It gives the right functional form; the exact $\alpha$ is tuned empirically.

---

## 6. Subsequent Work (Brief)

- **SD3.5**: QK-norm added (enabling stable 8B training), dual attention layers per MMDiT block (or image-only self-attention in Medium variant, MMDiT-X). Same text encoders and VAE.
- **Flux.1**: hybrid of 19 dual-stream (MMDiT) + 38 single-stream blocks with shared weights. Replaces sinusoidal positions with RoPE. Drops CLIP-G. Adds guidance distillation. Dual→single stream progression: early layers need modality-specialized experts; later layers share capacity once representations have aligned.