---
aliases:
  - ViT
Arxiv: https://arxiv.org/abs/2010.11929
date: 2021-10-26
---

A very very simple adaption of transformer, by dividing image to patches, add class label, add learnable 1d embedding, and go.

[Blog link: Transformers for Image Recognition at Scale](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)
[Paper link](https://arxiv.org/abs/2010.11929)

![[vit.gif]]

The whole model can be written [in 124 lines with PyTorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py).

What is the class token here? It's a random initalized variable that we append to the input, and it's transformed output is the only thing we care to produce the predicted category. Why don't we just use a normal FC layer? That's covered in the Appendix [[pdf/vit.pdf#page=16 |D.3]]. TL;DR is that we can do that, but we chose not to do that.

## Spatial Structure Preservation

A subtle but important property: ViT preserves the correspondence between token position and image patch position throughout the network. After patchification, each token has a fixed semantic meaning — "the patch at row $i$, column $j$" — and this never changes.

Why this holds:

- **Attention reweights values, it doesn't shuffle positions.** Token $i$ after attention is a weighted combination of all tokens' values, but it still lives at position $i$ in the output sequence.
- **The MLP after attention is pointwise.** It applies the same two-layer network to each token independently — no cross-token mixing.

So the sequence can always be reshaped back to a 2D spatial grid, just like a CNN feature map. This makes ViT a drop-in replacement for a CNN encoder in any task that needs spatially-grounded features — deformable attention lookups ([[TriPlane Tokenizer]], [[Deformable DETR]]), feature pyramids, segmentation. A global bottleneck like a VAE latent vector would destroy this property and break those uses.

(Caveat: the class token has no spatial position — strip it before reshaping.)