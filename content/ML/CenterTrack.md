---
share: true
Arxiv: https://arxiv.org/abs/2004.01177
pdf: "[[pdf/tracking_objects_as_points.pdf|tracking_objects_as_points]]"
original title: Tracking Objects as Points
---
Tracking object with [[CenterNet (Objects as Points)|CenterNet]], with very small modifications to the original work.

> our two main technical contributions: tracking-conditioned detection (Section 4.1) and offset prediction (Section 4.2)

[[ML/papers/pdf/tracking_objects_as_points.pdf#page=12&selection=114,15,115,49|tracking_objects_as_points, page 12]]

The intro of the paper summarize it pretty well:
> We condition the detector on two consecutive frames, as well as a heatmap of prior tracklets, represented as points. We train the detector to also output an offset vector from the current object center to its center in the previous frame. We learn this offset as an attribute of the center point at little additional computational cost. A greedy matching, based solely on the distance be- tween this predicted offset and the detected center point in the previous frame, suffices for object association. The tracker is end-to-end trainable and differentiable.

[[ML/papers/pdf/tracking_objects_as_points.pdf#page=2&selection=10,59,16,79|tracking_objects_as_points, page 2]]

![[center_track_overview.png]]

Pay attention to the heatmap here. Just like [[CenterNet (Objects as Points)|CenterNet]], this is Gaussian splatted keypoints. In training time, this come from ground truth, with augmentation. In inference time, this come from model output from $t-1$, very convenient. This can be done because the heatmap both capture model internal state, and is human constructible.
The augmentation is needed so that ground truth label may look like wacky model output. They [[ML/papers/pdf/tracking_objects_as_points.pdf#page=7&selection=53,0,142,65|added in jitter, false positives and false negatives]]. With this, the model can be trained with only static image also too. Just shift the image. See [[ML/papers/pdf/tracking_objects_as_points.pdf#page=7&selection=190,0,192,29|tracking_objects_as_points, page 7]].

The offset prediction is used for association. There's [[ML/papers/pdf/tracking_objects_as_points.pdf#page=12&selection=118,0,133,33|ablation study]] proving both it and heatmap as input is important.

The paper compare itself to Kalman filter and optical flow based methods. See [[ML/papers/pdf/tracking_objects_as_points.pdf#page=13&selection=216,0,218,39|tracking_objects_as_points, page 13]]. 

> On the high-framerate MOT17 dataset, any motion model suffices, and even no motion model at all performs competitively. On KITTI and nuScenes, where the intra-frame motions are non-trivial, the hand-crafted motion rule of the Kalman filter performs significantly worse, and even the performance of optical flow degrades. This emphasizes that our offset model does more than just motion estimation. CenterTrack is conditioned on prior detections and can learn to snap offset predictions to exactly those prior detections. Our training procedure strongly encourages this through heavy data augmentation.

[[ML/papers/pdf/tracking_objects_as_points.pdf#page=14&selection=7,0,14,76|tracking_objects_as_points, page 14]]