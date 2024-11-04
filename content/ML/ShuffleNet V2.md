---
aliases: 
Arxiv: https://arxiv.org/abs/1807.11164
pdf: "[[shufflenet_v2.pdf]]"
original title: "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
date: 2024-10-30
tags: 
year: 2018
---
Here I focus on the guidelines provided by the paper, instead of the new model.
Networks like [[MobileNet]] claims that they are fast, but not really. Their FLOPs are low and number of parameter low, but that doesn't really mean fast training / inference.

> The discrepancy between the indirect (FLOPs) and direct (speed) metrics can be attributed to two main reasons. First, several important factors that have considerable affection on speed are not taken into account by FLOPs. One such factor is memory access cost (MAC). Such cost constitutes a large portion of runtime in certain operations like group convolution. It could be bottleneck on devices with strong computing power, e.g., GPUs. This cost should not be simply ignored during network architecture design. Another one is degree of parallelism. A model with high degree of parallelism could be much faster than another one with low degree of parallelism, under the same FLOPs. 
> 
> Second, operations with the same FLOPs could have different running time, depending on the platform.

[[shufflenet_v2.pdf#page=2&selection=36,0,53,26|shufflenet_v2, page 2]]

GPU and ARM accelerators are quite different. MobileNet seems to be opimizing based on ARM. For all these guidelines below, we see larger effect on GPU compared with ARM.

### G1 Equal channel width minimizes memory access cost (MAC)

> MAC has a lower bound given by FLOPs. It reaches the lower bound when the numbers of input and output channels are equal.

That's basically answering "if the FLOP is fixed, how shall I design stages."

### G2 Excessive group convolution increases MAC

Again, same FLOPs, the more group there is, the higher MAC is.

> Therefore, we suggest that the group number should be carefully chosen based on the target platform and task. It is unwise to use a large group number simply because this may enable using more channels, because the benefit of accuracy increase can easily be outweighed by the rapidly increasing computational cost.

### G3 Network fragmentation reduces degree of parallelism

The more path / fragment there is, the slower the model is. 

### G4 Element-wise operations are non-negligible

ReLU, AddTensor etc. have heavy MAC. Remove ReLU and shortcut operations from ResNet, we observe around 20% speedup.