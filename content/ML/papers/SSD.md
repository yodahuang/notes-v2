---
share: true
---
[Arxiv link](https://arxiv.org/abs/1512.02325)

SSD stands for "Single Shot MultiBox Detector".

## Core idea

Generate a bunch of boxes to look at at different scale. For each of them, output class confidence + offset to the shape of the default box. Compute the loss by matching each box with ground truth box and do regression.

The classifier is a 3*3 conv kernel.

![[ssd_anchors.png|ssd_anchors.png]]

![[ssd_overview.png|ssd_overview.png]]