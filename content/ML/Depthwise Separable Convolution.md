---
date: 2024-10-19
---
> Depthwise conv has been popular- ized by [[MobileNet]] and Xception. We note that depthwise convolution is similar to the weighted sum op- eration in self-attention, which operates on a per-channel basis, i.e., only mixing information in the spatial dimension
[[ML/papers/pdf/convnext.pdf#page=4&selection=45,24,52,51|convnext, page 4]]

![[depth_wise_conv.png]]
Instead of having $D_{K}^2 * M * N * D_{F}^2$, where $K$ stands for kernel and $F$ stands for feature map, depth wise conv makes it two step, each step handling less capacity:
$$
D_{K}^{2}* M * D_{F}^{2} + M * N * D_{F}^2
$$
