---
Arxiv: https://arxiv.org/abs/1801.04381
pdf: "[[mobilenet_v2.pdf]]"
original title: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
date: 2024-10-12
tags:
---
Obviously a follow up to [[MobileNet]]. The new trick, as the title suggests, is [[inverted residuals]] and [[linear bottleneck]].

![[mobilenetv2_blocks.png]]
![[mobilenetv2_comparison.png]]

Take a closer look at how $n$ is chosen, and why $t$ in the first block is $1$. They are picked to make the network have more layers with lower feature map size. 

![[mobilenet_v2.pdf]]
