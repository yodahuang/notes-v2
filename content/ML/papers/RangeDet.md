---
Created: 2021-04-30T14:17
Link: https://arxiv.org/abs/2103.10039
share: true
---
Full range view detection.

### Motivation

Three problems with range view:

- Scale Variation between nearby and far away objects
    - [[FPN]] to rescue. Associate label with prediction by range.
    - The paper call it Range Conditioned Pyramid (RCP)
- Convolution is done in 2D range image, but the output we want is in 3D space.
    - Meta kernel that takes 3D Cartesian information in too (key important bit)
- Utilize compact 2D range view
    - Weighted Non-maximum suppression (not so interesting)

![[rangedet_1.png|rangedet_1.png]]

### Deep Dive

Meta Kernel: TL;DR:

I want to put Cartesian info into feature, but that’s hard. So I put that into kernel weight. That’s too expensive, so I steal some idea from MobileNet.

![[rangedet_2.png|rangedet_2.png]]

Let’s recall a normal CNN kernel, which is of size $h_k * w_k * c_{in} * c_{out}$﻿. The weight sharing mechanism means if a kernel is of $3* 3$﻿, then it just has 9 unique value per input channel. See here for a quick review of how normal CNN work:

> [!info] CS231n Convolutional Neural Networks for Visual Recognition  
> Table of Contents: Convolutional Neural Networks are very similar to ordinary Neural Networks from the previous chapter: they are made up of neurons that have learnable weights and biases.  
> [https://cs231n.github.io/convolutional-networks/](https://cs231n.github.io/convolutional-networks/)  

As we want to take Cartesian information in convolution, it need to be part of the feature. We can simply incorporate that into kernel weight. Thus, we need to relax the weight sharing, since the weight should change based on the Cartesian coordinates. The actual weight is calculated by a MLP.

Notice the strange element wise product there. This is because it use [depth-wise convolution](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) here, similar as MobileNet. By first doing per-channel convolution, then using $1*1$﻿ convolution to expand to output channel, it saves computation power.

[[Quotes]]