---
aliases:
  - ViT
share: true
---

A very very simple adaption of transformer, by dividing image to patches, add class label, add learnable 1d embedding, and go.

[Blog link: Transformers for Image Recognition at Scale](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)
[Paper link](https://arxiv.org/abs/2010.11929)

![[vit.gif|vit.gif]]

The whole model can be written [in 124 lines with PyTorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py).

What is the class token here? It's a random initalized variable that we append to the input, and it's transformed output is the only thing we care to produce the predicted category. Why don't we just use a normal FC layer? That's covered in the Appendix [[pdf/vit.pdf#page=16 |D.3]]. TL;DR is that we can do that, but we chose not to do that. 