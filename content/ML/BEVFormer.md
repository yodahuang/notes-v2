---
share: true
---
BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers
- [Arxiv link](https://arxiv.org/abs/2203.17270)
- [[pdf/bevformer.pdf|PDF link]]
- [Zhihu link (I know, why would anyone use Zhihu?)]()

The network's core representation is in BEV view (BEV query), it uses temporal self attention + spatial cross attention. The author states that they are inspired by Tesla's video spatial transformer.
![[bev_former_overview.png]]
![[bevformer_arch.png]]

## BEV Queries
A grid-shaped learnable parameters $Q \in \mathbb{R}^{H\times W \times C}$ , each cell correspond to real world geometry.
## Spatial Cross-Attetion
This part is like [[BEV baseline]], but it does not just grab context from CNN embedding. Instead, it attend on multiple cameras. It's not global attention. Instead, it uses [[Deformable Attention]], only interaction with small regions of interest. The reference points are sampled from the lifted pillar. 
$$
SCA(Q_{P,}F_{t)}= \frac{1}{|\mathcal{V}_{\text{hit}}|} \sum_{i\in\mathcal{V_{\text{hit}}}}\sum\limits^{N_{ref}}_{j=1}\text{DeformAttn}(Q_{P}, \mathcal{P}(p, i, j), F^i_t)
$$
Basically this means: sum all the things that's the attention between our BEV query and  the image feature around the projected selected points.

## Temporal Self-Attention
Align $B_{t-1}$ to $Q$ according to ego-motion, then do self-attention (also deformed) to ${Q, B'_{t-1}}$.

## How well does it perform?
Better than other camera-only method that time, but way worse than [[PointPillars]], and even worse than [[BEV baseline]].
