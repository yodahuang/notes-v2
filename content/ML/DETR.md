---
date: 2022-08-15
updated: 2026-05-23
year: 2020
Arxiv: https://arxiv.org/abs/2005.12872
pdf: "[[detr.pdf]]"
original title: End-to-End Object Detection with Transformers
---
---

Note original written by me and updated by Claude Sonnet 4.6 when revisiting

---

## Overview

DETR (DEtection TRansformer) reframes object detection as a **direct set prediction problem**, eliminating the hand-crafted pipeline components that dominate traditional detectors — no anchor generation, no NMS, no proposal networks. Two ingredients make this work:

> [!abstract] Key ingredients
> 1. **Bipartite matching loss** — a Hungarian-algorithm-based loss that uniquely assigns each prediction to exactly one ground truth object, making training permutation-invariant and duplicate-free by construction
> 2. **Transformer encoder-decoder** — the encoder builds a global image representation; the decoder takes a fixed set of learned *object queries* and attends to the encoder output to produce all detections in a single parallel pass

**Architecture pipeline:**

$$\text{Image} \xrightarrow{\text{CNN backbone}} \text{feature map} \xrightarrow{\text{1×1 conv}} \xrightarrow{\text{+ pos. enc.}} \text{Transformer Encoder} \xrightarrow{\text{+ object queries}} \text{Transformer Decoder} \xrightarrow{\text{FFN}} N \times (\text{class},\ \text{box})$$

**vs. traditional detectors:** [[Faster-RCNN]] predicts delta offsets from hand-crafted anchor boxes at thousands of spatial positions, then collapses near-duplicate predictions with NMS. DETR predicts absolute box coordinates directly from 100 learned slots, and deduplication is an emergent property of the loss and self-attention — no post-processing required.

**Performance on COCO:** Comparable AP to a well-tuned Faster R-CNN. Significantly better on large objects (global attention helps); worse on small objects (a known limitation). Requires much longer training (~500 epochs vs. 36 for Faster R-CNN). Inference is ~2× slower.

![[detr.png]]

## Object detection set prediction loss

DETR outputs a fixed set of $N$ predictions in one pass (typically $N = 100$, much larger than the number of objects in any image). Training requires matching predictions to ground truth in a way that is permutation-invariant and discourages duplicates.

### Step 1: Bipartite Matching

Pad the ground truth to size $N$ with $\varnothing$ (no object), then find the optimal one-to-one assignment:

$$
\hat{\sigma} = \mathop{\arg\min}_{\sigma \in \mathfrak{S}_N} \sum^N_i\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})
$$

The matching cost uses **raw probability** (not log) for the class term so it stays numerically commensurable with the box term:

$$\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\mathbb{1}_{c_i \neq \varnothing}\hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{c_i \neq \varnothing}\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$$

Solved efficiently with the Hungarian algorithm.

> [!info] Assuming best intention
> The matching finds the assignment most favorable to the model — each prediction is paired with whichever ground truth it best explains. Only after fixing this assignment do we compute the real gradient signal.

### Step 2: Hungarian Loss

With $\hat{\sigma}$ fixed, compute the actual loss over all $N$ pairs:

$$
\mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \sum^N_{i=1}\left[-\log\hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{c_i\ne \emptyset}\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)})\right]
$$

The class term switches to **log-probability** (standard cross-entropy) for the proper gradient. Per slot:

- **Matched to a real object** ($c_i \neq \varnothing$): cross-entropy on class + box loss (L1 + GIoU)
- **Matched to $\varnothing$**: cross-entropy toward "no object" only — the $\mathbb{1}_{c_i \neq \varnothing}$ indicator zeros out the box loss

> [!warning] Class imbalance
> With $N=100$ and say 7 real objects, 93 slots predict $\varnothing$. Without correction, background terms swamp the foreground loss. DETR down-weights the $\varnothing$ cross-entropy by $\times 0.1$ — the same problem [[Faster R-CNN]] solves by subsampling proposals, just done differently.

### Duplicates and misses

> [!tip] Duplicates: handled by construction
> Hungarian matching is one-to-one. If two slots both predict the same dog, only one is matched to it — the other is matched to padded $\varnothing$ and penalized for not predicting "no object." Self-attention among queries lets slots coordinate to avoid this fate during training.

> [!tip] Missing detections: no special penalty needed
> Every ground truth must match to *some* slot. If the model ignores an object, that object is matched to the least-bad slot, which then receives the full class + box loss, pulling it toward the correct answer.

## Object Queries: Learned Detection Slots

The paper calls these "learnt positional encodings," but that framing is misleading. A better mental model:

- The decoder is permutation-invariant — if you fed the same vector $N$ times, you'd get $N$ identical outputs. The $N$ object queries are $N$ *different* learned parameters that break this symmetry, giving each decoder slot a distinct starting point.
- They are **not** spatial anchors. Unlike [[SSD]] or [[Faster R-CNN]] anchors (hand-crafted grids at fixed positions, scales, aspect ratios), object queries have no hardcoded spatial meaning. Their specialization is entirely emergent from training.
- Figure 7 of the paper shows what they actually learn: each slot develops a soft bias toward certain regions and box sizes — some prefer upper-left, some prefer large image-spanning boxes — but this is a trained prior, not a designed one. There is no strong class-specialization per slot either (the paper verifies this with the 24-giraffe generalization experiment).
- They are added to the decoder input at **every attention layer** (both self-attention among queries and cross-attention to the encoder output). Through **cross-attention**, each query reads the encoder's global image representation to figure out what object to claim. Through **self-attention** among the 100 queries, each slot sees what the other slots are already claiming — this is what prevents duplicate detections, replacing NMS.
![[detr_figure7.png]]

> [!tip] Why "object queries," not "positional embeddings"
> The paper's own ablation confirms they are load-bearing: output positional encodings (object queries) cannot be removed, while spatial positional encodings in the *encoder* can be dropped with only a 1.3 AP drop. Their spatial/size biases are entirely emergent, not designed — the "positional" framing is misleading.

## Final Prediction: Direct Regression

Each of the $N$ decoder output embeddings passes through a **shared FFN** (3-layer MLP with ReLU, hidden dim $d$), producing two heads independently:

1. **Bounding box** — $(c_x, c_y, w, h)$: center x, center y, width, height, all normalized to $[0, 1]$ relative to image dimensions. Sigmoid-activated (see `.sigmoid()` in the PyTorch code). No anchors, no delta offsets — direct absolute coordinates.
2. **Class** — linear layer + softmax over $(N_{\text{classes}} + 1)$ logits, where the $+1$ is the $\varnothing$ ("no object") class, playing the role of background.

The box loss combines two terms:

$$\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) = \lambda_{\text{iou}}\mathcal{L}_{\text{iou}}(b_i, \hat{b}_{\hat{\sigma}(i)}) + \lambda_{\text{L1}}||b_i - \hat{b}_{\hat{\sigma}(i)}||_1$$

> [!info] Why L1, not L2?
> L2 (squared error) penalizes large errors quadratically — early in training when predictions are wildly off, this produces huge gradients and makes training unstable. L1 grows linearly, which is more robust to these early outliers. Note this is different from Faster R-CNN, which uses Smooth L1 (Huber) on anchor *offsets* — those are normalized by anchor size, so the scale problem is already partially handled. DETR predicts absolute coordinates in $[0, 1]$, so robustness to large errors matters more.

> [!info] Why GIoU, not plain IoU?
> Standard IoU $= \frac{|A \cap B|}{|A \cup B|}$ is zero whenever two boxes don't overlap at all — regardless of how far apart they are. This means **zero gradient** for any non-overlapping prediction early in training, which is most of them. GIoU fixes this by adding a penalty term:
>
> $$\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}$$
>
> where $C$ is the smallest axis-aligned box enclosing both $A$ and $B$. The second term measures how much of the enclosing box is wasted (not covered by either box) — it's large when the boxes are far apart and shrinks to zero as they converge. GIoU $\in [-1, 1]$ (vs IoU $\in [0, 1]$), so it always provides a gradient signal. The loss is $\mathcal{L}_{\text{iou}} = 1 - \text{GIoU}$.

> [!warning] Why both together?
> L1 gives a coordinate-space gradient but is not scale-invariant — a 10px error on a tiny box is much worse than on a large box, but L1 treats them equally. GIoU is scale-invariant but operates in IoU-space and has weak signal when boxes are nearly aligned. Together they complement each other: GIoU handles scale and overlap geometry; L1 provides stable coordinate-space gradients throughout training.

## Parallel Decoding vs. Autoregressive

Prior set-prediction work (cited in the paper) used autoregressive RNNs: predict one box at a time, conditioning each on all previous outputs — the same spirit as modern LLMs generating tokens. This naturally avoids duplicates but has two problems: (a) inference is sequential — $O(N)$ steps; (b) bounding boxes have no natural ordering, so an arbitrary one must be imposed, making the learning problem harder.

DETR's parallel decoding solves both. The self-attention among the $N$ queries plays the same role as autoregressive conditioning — every slot sees what every other slot is "thinking" — but all within a single forward pass across decoder layers. The Hungarian matching loss enables permutation-invariant training with no canonical output order required.

> [!info] Could DETR be decoder-only?
> Yes — the encoder is important but not essential. Table 2 of the paper shows 0 encoder layers achieves 36.7 AP vs. 40.6 with 6 layers (−3.9 AP). The encoder's job is to globally "pre-separate" instances before the decoder focuses each query. Without it, the decoder must do both jobs and performance suffers, particularly on large objects.

**The fundamental tradeoff:**

| | Autoregressive (RNN prior work) | DETR (parallel) |
|---|---|---|
| Inference cost | $O(N)$ sequential steps | $O(1)$ — all slots at once |
| Output count | Variable, stop naturally | Fixed $N$ (most slots → $\varnothing$) |
| Deduplication | Free via conditioning | Self-attention + set loss |
| Output ordering | Must be defined | Permutation-invariant by design |

## PyTorch inference code
```python

import torch  
from torch import nn  
from torchvision.models import resnet50  

class DETR(nn.Module):  
	def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
		super().__init__()  
		# We take only convolutional layers from ResNet-50 model  
		self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])  
		self.conv = nn.Conv2d(2048, hidden_dim, 1)  
		self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)  
		self.linear_class = nn.Linear(hidden_dim, num_classes + 1)  
		self.linear_bbox = nn.Linear(hidden_dim, 4)  
		self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))  
		self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))  
		self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))  
		
	def forward(self, inputs):  
		x = self.backbone(inputs)  
		h = self.conv(x)  
		H, W = h.shape[-2:]  
		pos = torch.cat([  
			self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),  
			self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),  
		], dim=-1).flatten(0, 1).unsqueeze(1)  
		h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),  
		self.query_pos.unsqueeze(1))  
		return self.linear_class(h), self.linear_bbox(h).sigmoid()  

detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)  
detr.eval()  
inputs = torch.randn(1, 3, 800, 1200)  
logits, bboxes = detr(inputs)
```
For clarity it uses learnt positional encodings in the encoder instead of fixed, and positional encodings are added to the input only instead of at each transformer layer. Making these changes requires going beyond PyTorch implementation of transformers, which hampers readability.
