---
Created: 2021-10-24T15:36
Link: https://arxiv.org/abs/1612.00593
share: true
---
### Summary

Use Max Pooling to get unordered global feature. If per-point segmentation is needed, concat it with local point feature and voila!

### Properties of Point Sets in $\mathbb{R}^n$﻿

- Unordered. It doesn't matter which point comes first. PointNet use a big max pooling to make this happen.
- Interaction among points. It need to capture local structures from nearby points. I don't see this part from PointNet.
- Invariance under transformations. Rotate the points and the result is still the same. Accomplished by having a mini PointNet (T-Net) output a rotational matrix, thus align them to a canonical space. Not much performance improvement though.

### Model structure

![[pointnet.png]]

Note that the dimension of the global pose feature greatly affect the performance of the model.

Intuitively, the network learns to summarize a shape by a sparse set of key points.