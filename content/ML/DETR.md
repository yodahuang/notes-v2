---
date: 2022-08-15
updated: 2026-05-23
year: 2020
Arxiv: https://arxiv.org/abs/2005.12872
pdf: "[[detr.pdf]]"
original title: End-to-End Object Detection with Transformers
---
---

Note original written by me and updated by Claude Sonnet 4.6 when revisiting, and then again updated by Fable 5 when revisiting again.

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

The paper calls these "learnt positional encodings," and that framing is exactly right mechanically — but the decoder query is really **two separate tensors** that are easy to conflate:

- **Decoder embedding $X$** — the content part. This is just the decoder's ordinary residual stream (self-attn → cross-attn → FFN with residual adds). It is **initialized to zero** (Appendix A.3: *"the decoder receives queries (initially set to zero), output positional encoding (object queries), and encoder memory"*) — there is no autoregressive input token to seed it, so content is built up purely by attending to the image. After the first cross-attention, $X$ is a weighted combination of encoder features, i.e. it lives in the same space as image features.
- **Object query $P$** — the positional part. A fixed learned parameter per slot (100 of them), the same at every layer. It encodes *slot identity* the way sinusoidal PE encodes pixel location.

**How $P$ is injected** (Appendix A.1, Eq. 7, and Fig. 10): positional encodings are added to **Q and K only, never V** — at *every* attention layer, exactly like the encoder's spatial PE (Appendix A.4: the best variant adds sine PE at every attention layer):

| Attention | Q | K | V |
|---|---|---|---|
| Decoder self-attn | $X + P$ | $X + P$ | $X$ |
| Decoder cross-attn | $X + P$ | $\text{mem} + \text{spatial PE}$ | $\text{mem}$ |

where $\text{mem}$ is the encoder output. So "object query" = "positional encoding, but for slot identity instead of pixel location" — the paper's line *"similarly to the encoder, we add them to the input of each attention layer"* is saying it's the same trick, applied to break decoder permutation symmetry.

![[detr_decoder_query_composition.svg|690]]

Why the symmetry-breaking matters: the decoder is permutation-invariant — if you fed the same vector $N$ times, you'd get $N$ identical outputs. The $N$ distinct $P$ vectors give each slot a different starting point. Through **cross-attention**, each query reads the encoder's global image representation to figure out what object to claim; through **self-attention** among the 100 queries, each slot sees what the others are claiming — this is what prevents duplicates, replacing NMS.

> [!note] "Not spatial anchors" — interpretation, not a paper quote
> The paper never says this verbatim; it's a synthesis supported by two pieces of evidence. Fig. 7: each slot learns several *modes* (soft biases toward regions and box sizes, nearly all slots also having a mode for large image-spanning boxes) — messier and more overlapping than a hand-designed anchor grid like [[SSD]] or [[Faster R-CNN]], where each anchor has one fixed scale/aspect/location. And the 24-giraffe generalization experiment, which the paper says *"confirms that there is no strong class-specialization in each object query."* The specialization is a trained prior, not a designed one. ([[DAB-DETR]] later reframes $P$ as explicit anchor-box coordinates precisely because it behaves like a positional encoding.)

![[detr_figure7.png]]

> [!tip] Load-bearing, unlike encoder PE
> The paper's ablation shows object queries cannot be removed, while the *encoder's* spatial positional encodings can be dropped with only a 1.3 AP loss. Mechanically they are positional encodings (added to Q/K at every layer); what they encode — slot identity with emergent, not designed, spatial biases — is what earns the different name.

> [!info] The clearest picture of this $X$ vs $P$ split is in [[DAB-DETR]]
> DAB-DETR's Fig. 2 draws exactly the decomposition above — encoder self-attention (query = image feature + spatial PE) vs. decoder cross-attention (query = decoder embedding $X$ + learnable query $P$), highlighting in purple that the *only* structural difference is the query. Its Fig. 8 then places DETR (panel a) beside every follow-up that reworks $P$, and argues $P$ is the root cause of DETR's slow convergence.

![[detr_encoder_vs_decoder_attn.png]]

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

> [!warning] This snippet is not DETR's real decoder
> `tgt = self.query_pos.unsqueeze(1)` — the object queries are fed in *as* the decoder input, once. There is no separate zero-initialized decoder embedding, and stock `nn.Transformer` never re-adds anything at later layers. So the snippet conflates $X$ (content, init 0) and $P$ (object queries, re-added to Q/K at every layer) into a single tensor. The real implementation passes `tgt = torch.zeros_like(query_embed)` and sums `query_pos` into Q and K inside every attention layer.
