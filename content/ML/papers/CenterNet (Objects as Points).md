---
aliases:
  - CenterNet
share: true
Arxiv: https://arxiv.org/abs/1904.07850
pdf: "[[ML/papers/pdf/centernet_object_as_points.pdf|centernet_object_as_points]]"
original title: Objects as Points
---
Note: this and the [[CenterNet (Keypoint Triplets)|centernet_triplets]] paper share the same network name.

Unlike the [[CenterNet (Keypoint Triplets)|centernet_triplets]] paper, this one is much straight forward, no [[CenterNet (Keypoint Triplets)#Center pooling and Cascade corner pooling|center pooling or cascade corner pooling]].  It only uses a single point, the center of object as anchor, and predict all the other stuff from it.
## Loss
The final layer of the model is a heatmap with output stride of $R=4$.
Ground truth center is splatted with Gaussian kernel (so that pixels around it is also fine), and then compare with the heatmap with Focal loss. See [[ML/papers/pdf/centernet_object_as_points.pdf#page=3&selection=166,0,259,37|centernet_object_as_points, page 3]]. A local offset is also predicted to recover the discretization error.
$$
L_{\text {det }}=L_k+\lambda_{size} L_{size}+\lambda_{off} L_{off}
$$
Here $size$  is the bounding box params, and $off$ is local offset. The values used in the experiments are $\lambda_{size} = 0.1$ and $\lambda_{off} = 1$.
## From center to object
No NMS here. As long as point is a global maxima, we say it's a center.
## 3D detection
> However, depth is difficult to regress to directly. We instead use the output transformation of Eigen et al. [13] and d = 1/σ( ˆd) − 1, where σ is the sigmoid function. 

[[ML/papers/pdf/centernet_object_as_points.pdf#page=4&selection=149,0,173,1|centernet_object_as_points, page 4]]

Orientation prediction is more elaborate.
> We use an 8-scalar encoding to ease learning. The 8 scalars are divided into two groups, each for an angular bin. One bin is for angles in $B_1 = [− \frac{7\pi}{6}, \frac{\pi}{6}]$ and the other is for angles in $B_2 = [− \frac{\pi}{6}, \frac{7\pi}{6}]$. Thus we have 4 scalars for each bin. Within each bin, 2 of the scalars $b_{i}\in R^2$  are used for softmax classification (if the orientation falls into to this bin i). And the rest 2 scalars $a_{i}\in R^2$ are for the sin and cos value of in-bin offset (to the bin center $m_i$).

And then the loss is classification for $b_i$ and regression for $a_i$.

[[ML/papers/pdf/centernet_object_as_points.pdf#page=11&selection=378,0,487,30|centernet_object_as_points, page 11]]
## Human pose
How do you know which keypoints belong to the same human?
1. Add a output, offset from center to all the keypoints.
2. Refine the keypoints, by predicting a joint keymap (just like how we get the center)
3.  
> We then snap our initial predictions to the closest detected keypoint on this heatmap. Here, our center offset acts as a grouping cue, to assign individual keypoint detec- tions to their closest person instance.

[[ML/papers/pdf/centernet_object_as_points.pdf#page=4&selection=326,0,331,39|centernet_object_as_points, page 4]]

There's an extra experiments([[ML/papers/pdf/centernet_object_as_points.pdf#page=7&selection=329,0,337,55|centernet_object_as_points, page 7]]) showing that not predicting this heatmap makes things worse.

## Inference
They used augmentation in inference and then use bagging to output average. It's the first time I've seen this trick (is it even legal?)

> We use three levels of test augmentations: no augmentation, flip augmentation, and flip and multi-scale (0.5, 0.75, 1, 1.25, 1.5). For flip, we average the network outputs before decoding bounding boxes. For multi-scale, we use NMS to merge results. These augmentations yield different speed-accuracy trade-off, as is shown in the next section

[[ML/papers/pdf/centernet_object_as_points.pdf#page=5&selection=257,0,263,7|centernet_object_as_points, page 5]]