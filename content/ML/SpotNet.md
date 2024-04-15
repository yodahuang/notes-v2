---
share: true
---
# SpotNet: Self-Attention Multi-Task Network for Object Detection
- [Arxiv link](https://arxiv.org/abs/2002.05540)
- [[pdf/spotnet.pdf|pdf]]

SpotNet is an improvement on [[CenterNet (Keypoint Triplets)]], which itself is improved based on [[CornerNet]]. It's just like RCNN, Fast RCNN and Faster RCNN, but they are not from the same author / group.
![[spotnet_overview.png]]

It basically adds two idea:
- Multi-task learning. By asking the model to do both segmentation and bounding box detection, the sahred parameters are more generic and less prone to overfitting.
- Self-Attention. As shown in the image, the result of the segmentation (before binarization) is used as a mask to be applied to the bounding box detection task. This is actually very clever.
Q: Where do you get the label for the segmentation task? A: use video and get lower-quality-but-good-enough label by traditional CV method.