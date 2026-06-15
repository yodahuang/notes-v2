---
date: 2026-06-14
pdf: "[[d4rt.pdf]]"
aliases:
  - D4RT
year: 2026
Arxiv: https://d4rt-paper.github.io/
original title: "Efficiently Reconstructing Dynamic Scenes One D4RT at a Time"
---

---
*Notes written by Claude (Opus 4.8) from a reading discussion.*

---

D4RT (CVPR 2026 best paper) does **4D reconstruction from a single video** — depth, point clouds, 3D point tracks, and camera parameters for *dynamic* scenes, where moving objects break the rigid-world assumption most 3D reconstruction relies on.

Its one new idea is a **unified query interface**: every task becomes the same question — *"where in 3D is 2D point $(u,v)$ from source frame $t_\text{src}$, seen at target time $t_\text{tgt}$, in camera $t_\text{cam}$'s frame?"* — answered by a decoder cross-attending into a frozen scene representation. That reframing is what lets one model do **dynamic** correspondence (which [[VGGT]] structurally cannot) and unlocks an efficient dense-tracking algorithm. The rest is recombination: a [[VGGT]] encoder (initialized from **VideoMAE** — load-bearing, see [[#What the results lean on]]) feeding an [[#It's VGGT + SRT|SRT]]-style cross-attention decoder.

![[d4rt-overview.png]]

## Everything is one query

The query $\mathbf{q} = (u, v, t_\text{src}, t_\text{tgt}, t_\text{cam})$ is decoded independently against the frozen scene representation $F$ into a 3D point $\mathbf{P} = \mathcal{D}(\mathbf{q}, F)$. The three temporal indices are **free, independent parameters** — so every 4D task is just a slice of the Cartesian product:

| Task            | vary                                  | fix                                                   |
| --------------- | ------------------------------------- | ----------------------------------------------------- |
| **Point track** | $t_\text{tgt}=t_\text{cam}=1\ldots T$ | one $(u,v,t_\text{src})$                              |
| **Point cloud** | all $(u,v)$, all $t_\text{src}$       | $t_\text{cam}$ fixed (one world frame)                |
| **Depth map**   | all $(u,v)$                           | $t_\text{src}=t_\text{tgt}=t_\text{cam}$, keep only Z |
| **Extrinsics**  | $(h,w)$ grid, two $t_\text{cam}$      | → Umeyama between the two point sets                  |
| **Intrinsics**  | $(h,w)$ grid                          | pinhole equation on predicted points                  |

Depth is the *identity* case ($t_\text{src}=t_\text{tgt}=t_\text{cam}$, keep Z); point cloud predicts every pixel directly in one shared frame, sidestepping the noisy per-frame stitching that pairwise methods ([[DUSt3R]] lineage) need.

## Why the query works

**Dynamic correspondence** is the capability no dense decoder has. VGGT's tracking assumes the same 3D point across views (rigid world) — a moving leg breaks it. D4RT's query has time built in, so "where did this pixel *go*" is one forward pass; vary $t_\text{tgt}$ for the whole trajectory.

The decoder answers each query **independently**, and that buys three things: cheap sparse training (~2048 queries/clip vs. dense $O(THW)$), free add/drop of queries (what makes [[#Dense tracking as a coverage problem|Algorithm 1]] possible), and — crucially — a stronger encoder. They found query-to-query attention *hurts*: forcing every query to be answered cold from $F$ alone pushes all scene understanding into the encoder. It's the mirror image of VGGT's [[VGGT#Over-complete prediction: the most interesting design choice|over-complete prediction]] — both make the shared representation carry the full geometric burden.

## It's VGGT + SRT

The encoder is VGGT unchanged (ViT-g, interleaved frame-wise + global attention over spatio-temporal patches), initialized from VideoMAE. The encode-then-query *framing* — a "Global Scene Representation" you cross-attend into — is **SRT** (Scene Representation Transformer, Sajjadi et al. 2022): SRT encodes posed images into latents and queries them with rays for color. D4RT swaps the ray query for a spatio-temporal point query and outputs 3D position instead of RGB.

> [!faq]+ Placing it in the NeRF → SRT → D4RT lineage
> Two axes, not just "more tasks":
> - **Richness of use:** SRT (one renderer) → VGGT (task-specific heads) → D4RT (one query interface subsuming all tasks).
> - **What forces geometry:** SRT supervises RGB (geometry stays implicit); D4RT supervises 3D points directly (geometry explicit from the start).
>
> On those axes: **SRT = implicit + appearance**, **[[TriPlane Tokenizer]] = explicit + appearance** (triplane, but trained by [[NeRF]]-style rendering), **D4RT = implicit + geometry**; VGGT sits between SRT and D4RT. Shared template throughout: *encode multi-view observations into a scene rep, then query it.*
> - **[[NeRF]]:** per-scene MLP, ray-marched. No generalization.
> - **SRT:** learned encoder + ray query for color. NeRF's task, feedforward.
> - **D4RT:** point query → 3D position, no rendering anywhere.
>
> One consequence of D4RT's *implicit* $F$: it has no spatial coordinates, so you can't look up a feature by 3D position (unlike a triplane) — the decoder is mandatory.

## Two design choices that carry the results

> [!note]+ Local RGB patch → sub-pixel detail
> Each query also gets a Fourier embedding of $(u,v)$ and the local **$9\times9$ RGB patch** at $(u,v)$ — the cheap analog of DPT's skip connections. Not minor: AbsRel(S) $0.366\to0.302$, ATE $0.173\to0.091$ (Table 7). The real payoff (App. C): since $(u,v)$ is continuous, decoding resolution decouples from encoding resolution — feed full-res patches while the encoder stays at $256^2$ and recover sub-pixel detail (hair, sharp boundaries; PDBE $3.32\to2.19$). VGGT's DPT head can't do this.

> [!note]+ Cameras for free (no camera head)
> Both recovered analytically from point predictions: **intrinsics** from the pinhole model on a decoded grid ($f_x=p_z(u-0.5)/p_x$, median-pooled); **extrinsics** by decoding the same grid at two $t_\text{cam}$ values and running **Umeyama** (3×3 SVD) between the two point sets. This is the *inverse* of VGGT's over-complete prediction: VGGT predicts cameras+depth+points jointly to force consistency; D4RT predicts only points and derives cameras by algebra. The point head *is* the camera head.

## Dense tracking as a coverage problem

Tracking every pixel naively is $O(T^2HW)$. Instead, a track from $(u,v,t_\text{src})$ marks **every spatio-temporal pixel it visibly passes through** as done in an occupancy grid $G$, so new tracks start only from unvisited pixels. Adaptive **5–15× speedup** (more for static scenes, which have more redundancy); 18–300× more full-video tracks than DELTA/SpatialTrackerV2 at fixed FPS (Table 3). Only possible because queries are independent and cheap.

```pseudo
\begin{algorithm}
\begin{algorithmic}
\REQUIRE Input video $V$, encoder $\mathcal{E}$, decoder $\mathcal{D}$
\STATE $F \gets \mathcal{E}(V)$ \COMMENT{Global Scene Representation}
\STATE $G \gets \{\textbf{false}\}^{T\times H\times W}$ \COMMENT{Occupancy grid}
\STATE $\mathcal{T} \gets \emptyset$ \COMMENT{Set of dense tracks}
\WHILE{$\textbf{any}(G = \textbf{false})$}
    \STATE Sample a batch $B$ of unvisited source points from $G$
    \FOR{each $(u,v,t_\text{src}) \in B$ \textbf{in parallel}}
        \STATE $Q \gets \{(u,v,t_\text{src}, t_\text{tgt}{=}k, t_\text{cam}{=}k)\}_{k=1}^{T}$
        \STATE $P \gets \{\mathcal{D}(q_k, F)\}_{k=1}^{T}$ \COMMENT{Run decoder}
        \STATE Mark $\textbf{Visible}(P)$ pixels as visited in $G$
        \STATE $\mathcal{T} \gets \mathcal{T} \cup P$
    \ENDFOR
\ENDWHILE
\RETURN $\mathcal{T}$
\end{algorithmic}
\end{algorithm}
```

## What the results lean on

> [!warning] The headline numbers are a VideoMAE fine-tune
> Without VideoMAE init the model **collapses** — ATE $0.091\to0.334$, AbsRel(S) $0.302\to0.738$ (Table 11). The VGGT+DINOv2 comparison is fair-ish (both lean on internet-scale pretraining), but VideoMAE is *video* pretraining vs. DINOv2's static images — a more task-appropriate foundation. Open question (same as the [[VGGT#One open question|VGGT note]], DINOv2 → DINOv3): how much of D4RT's edge is the query architecture vs. the backbone? Not controlled for.

> [!note] Smaller print
> - **Auxiliary losses help — and echo VGGT.** 2D position, normals, displacement, visibility, confidence each improve results (Table 8) — same mechanism as VGGT's over-complete prediction, but as linear projections discarded at inference. Trade-off: dropping the confidence loss slightly *helps* depth but wrecks pose.
> - **Confidence loss** $c\,\mathcal{L}_\text{3D}-\lambda\log c$ ⇒ optimum $c^\star\propto 1/\mathcal{L}_\text{3D}$: calibrated uncertainty learned jointly with accuracy (Kendall & Gal; see [[Uncertainty based learnable weighting]]). The 3D L1 is on depth-normalized points through $\text{sign}(x)\log(1+|x|)$ to stop far points dominating.
> - **Two unablated tricks:** 30% of queries oversampled near depth/motion boundaries (Sobel); $t_\text{tgt}=t_\text{cam}$ with prob 0.4 (the easy identity case). Stated as fact; likely matter for reproduction.
> - **Scale/training:** ViT-g encoder ≈1B params, decoder 144M (so "lightweight" = the decoder). Kauldron; 48-frame $256^2$ clips, 2048 queries; AdamW, 500k steps, 64 TPU chips, ~2 days. Training data includes **Waymo Open** (relevant for driving-scene comparisons vs. VGGT).
> - **Long videos (App. B):** overlapping segments stitched by Umeyama on high-confidence overlap points, no loop closure (unlike VGGT-Long). Best ATE on KITTI 1000-frame sequences.
