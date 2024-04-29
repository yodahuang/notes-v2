---
Arxiv: https://arxiv.org/abs/1903.05625
pdf: "[[tracktor.pdf]]"
original title: Tracking without bells and whistles
date: 2024-04-21
tags:
  - multi-object-tracking
  - "#tracking-by-detection"
---
A long time classic, often appear in the newer paper as "hey see how much my new method beats Tracktor" section. 

> We show that one can achieve state-of-the-art tracking results by training a neural network only on the task of detection.

[[tracktor.pdf#page=1&selection=77,5,83,0|tracktor, page 1]]

Using an off-the-shelf pre-trained [[Faster R-CNN]], it's possible to do good enough tracking.
![[tracktor_overview.png]]
> Several CNN-based detection algorithms [52, 63] contain some form of bounding box refinement through regression. We propose an exploitation of such a regressor for the task of tracking.

[[tracktor.pdf#page=3&selection=66,35,69,50|tracktor, page 3]]

> Bounding box regression. The first step, denoted with blue arrows, exploits the bounding box regression to extend active trajectories to the current frame t. This is achieved by regressing the bounding box $b^k_{t−1}$ of frame $t − 1$ to the object’s new position $b^k_t$ at frame t. In the case of Faster R-CNN, this corresponds to applying RoI pooling on the features of the current frame but with the previous bounding box coordinates.

[[tracktor.pdf#page=3&selection=248,0,285,20|tracktor, page 3]]

Note it's not saying "predicting next frame's detection based on this frame". That'll be [[tracking-by-regression]]. It's saying "hey it was here last frame, I bet it'll be here this frame too, with some small adjustment maybe?" 

Of course, one need to handle
- Track deletion if the new confidence is low (maybe for a while)
- Track proposal if it covers a potentially new object.

## Tracking extensions

There's the assumption of "the object is still roughly around its last location" for the whole thing to work. We can improve that by applying extra motion model to compensate for that. 

> For sequences with a moving camera, we apply a straightforward camera motion compensation (CMC) by aligning frames via image registration using the Enhanced Correlation Coefficient (ECC) maximization as introduced in [16]. For sequences with comparatively low frame rates, we apply a constant velocity assumption (CVA) for all objects as in [11, 2]

[[tracktor.pdf#page=4&selection=119,13,125,19|tracktor, page 4]]