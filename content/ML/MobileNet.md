---
Arxiv: https://arxiv.org/abs/1704.04861
pdf: "[[mobilenet_v1.pdf]]"
original title: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
date: 2024-10-12
tags:
---
The core idea of MobileNet is [[Depthwise Separable Convolution]].
Refer to the PDF for exact layer setup / network setup.

The model can be made smaller by two ways:
- Making it thinner (less channels)
- Reduce representation (shrink input resolution). Note in this paper they are just providing different input resolution, instead of having a downsample layer.
What they learned: 
- Making model thinner is better than shallower (with less layers, even if they do not operate on feature map of a different resolution). Table 5
- Width / resolution scaler is smooth, unless we make the network way too thin. Table 6, Table 7