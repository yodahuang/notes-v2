---
date: 2026-05-23
pdf: "[[deformable_detr.pdf]]"
aliases:
  - deformable attention
year: 2021
Arxiv: https://arxiv.org/abs/2010.04159
original title: "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
---
---

Note from my discussion with Claude Opus 4.7, drafted by Sonnet 4.6

---

Paper: [Deformable DETR (Zhu et al., 2020)](https://arxiv.org/abs/2010.04159), §4.1.

> [!tip] Core insight
> Instead of attending to every spatial location (cost $O(H^2W^2)$), each query learns **K spatial offsets** from a reference point and only attends to those K locations. Offsets and weights both come from the query alone — no query-key dot product.

## The formula (single-scale)

$$

 \text{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^{M} W_m \left[ \sum_{k=1}^{K} A_{mqk} \cdot W'_m \cdot x(p_q + \Delta p_{mqk}) \right] 

$$


![[deformattn_spatial_sampling.svg|637]]

| Symbol           | Meaning                                         |
| ---------------- | ----------------------------------------------- |
| $z_q$            | Query feature vector ($C$-dim)                  |
| $p_q$            | Reference point (2D coordinate)                 |
| $x$              | Feature map ($C \times H \times W$)             |
| $m$              | Head index ($M$ heads, default 8)               |
| $k$              | Sample index ($K$ per head, default 4)          |
| $\Delta p_{mqk}$ | Sampling offset (2D, unconstrained range)       |
| $A_{mqk}$        | Attention weight; $\sum_k A_{mqk} = 1$ per head |
| $W'_m$           | Value projection matrix (per head)              |
| $W_m$            | Output projection matrix (per head)             |

## How offsets and weights are computed

Both $\Delta p_{mqk}$ and $A_{mqk}$ come from a **single linear projection on $z_q$** with $3MK$ output channels:

- First $2MK$ channels → reshape to $(M, K, 2)$ → the offsets $\Delta p_{mqk}$. Unconstrained; can point anywhere in the image.
- Last $MK$ channels → softmax over $K$ per head → the weights $A_{mqk}$.

![[deformattn_pipeline.svg|637]]

 Both $\Delta p_{mqk}$ and $A_{mqk}$ branch from a single linear projection on $z_q$. No feature map keys are consulted.

> [!important] No keys involved
> Unlike [[Attention is all you need|Multi-Head Attention]], the weights are **not** computed from a query-key dot product. The query alone decides where to look and how much to weight each location. This is the defining departure from standard attention.

> [!note] Sparse in count, not in reach
> The constraint is on **how many** locations are sampled (K=4 per head), not **how far** the offsets can reach — $\Delta p_{mqk}$ is unconstrained in range. This is the key distinction from local-window attention (e.g. [[Swin Transformer]]) or CNNs, which restrict the radius. DeformAttn restricts the budget.

The sample locations $p_q + \Delta p_{mqk}$ are generally fractional coordinates, so the feature map is read via **bilinear interpolation** — same mechanism as [[Deformable Convolution]].

![[deformable_attn_alt.png]]

## Multi-scale extension

$$

 \text{MSDeformAttn}(z_q, \hat{p}_q, {x^l}) = \sum_{m=1}^{M} W_m \left[ \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot W'_m \cdot x^l(\phi_l(\hat{p}_q) + \Delta p_{mlqk}) \right] 

$$

Changes from single-scale:

- $L$ feature levels (e.g. $L=4$ from ResNet C3–C5 + one extra strided conv).
- Reference point $\hat{p}_q \in [0,1]^2$ is normalized; $\phi_l(\hat{p}_q)$ rescales it to pixel coords at level $l$.
- Attention weights $A_{mlqk}$ are now softmaxed over all $L \times K$ combinations per head.
- Each head samples $LK$ total locations, routing freely across scales.

## Encoder vs decoder usage

**Encoder** — each pixel is its own query and reference point. A learned scale-level embedding $e_l$ distinguishes which feature level each pixel comes from. FPN not used — multi-scale attention already exchanges cross-level information.

**Decoder cross-attention** — $N$ object queries; reference point $\hat{p}_q = \sigma(Wz_q)$ predicted from query embedding. Box predictions are relative offsets w.r.t. the reference point, not absolute coordinates (see A.3).

**Decoder self-attention** — standard attention, unchanged (only $N$ queries, cost is small).

## Decoder object queries and the reference point

In [[DETR]], the $N$ object queries carry no spatial information — specialization emerges implicitly through bipartite matching over hundreds of epochs. Deformable DETR makes spatial grounding **explicit**: the reference point $\hat{p}_q = \sigma(Wz_q)$ gives each query a 2D anchor from the start.

The two-stage variant pushes further — spatial grounding comes entirely from the encoder:

| Variant | Reference point source | Query $z_q$ source |
|---|---|---|
| [[DETR]] | none — fully implicit | learned embedding |
| Deformable DETR (1-stage) | $\sigma(W z_q)$ | learned embedding |
| Deformable DETR (2-stage) | encoder proposal center | encoder feature at proposal |

> [!note] The progression
> DETR trusts queries to figure out space implicitly. One-stage gives them an explicit anchor. Two-stage doesn't trust the queries at all — the "elegant slot" idea is quietly retired.

## Iterative bounding box refinement (§4.2)

Each of the $D=6$ decoder layers refines the box from the previous layer rather than predicting independently. The reference point for layer $d$ is the box center predicted by layer $d-1$, and sampling offsets are modulated by that box's predicted width/height — so the attention window shrinks as the box tightens. Detection heads are not shared across layers. See A.4 for the full formula and stop-gradient details.

## Relation to deformable convolution

Setting $L=1$, $K=1$, $W'_m = I$ recovers deformable convolution exactly. The lineage is [[Deformable Convolution]] → DeformAttn — **not** MHA with deformable sampling bolted on.

## When does DeformAttn apply?

DeformAttn is a **spatial-domain trick**. It has three implicit assumptions that all depend on the feature map being a spatially coherent conv grid:

1. Reference points are meaningful 2D coordinates in a continuous space.
2. Offsets $\Delta p_{mqk}$ can point anywhere in that space with real-valued precision.
3. Bilinear interpolation can retrieve features at fractional coordinates.

This is why it works when you combine attention with CNNs (conv feature maps are dense, spatially coherent, and continuously interpolable) and why it doesn't directly translate to a ViT — a ViT token at position $(3, 5)$ has no neighbor at $(3.7, 5.2)$; there is nothing to interpolate between.

The paper's 2020 publication date matters here: this is still the CNN+attention hybrid era. The implicit assumption that you have a real spatial feature map is never stated because it was universal.

> [!tip] Rule of thumb
> Use DeformAttn when (a) your keys live in a spatially structured feature map, and (b) you know the relevant information is spatially local to the query but don't know exactly where. If the feature map has no spatial meaning (flat ViT tokens, language sequences), the offset mechanism loses its grounding.
