---
aliases:
  - NeRF
share: true
Arxiv: https://arxiv.org/abs/2003.08934
pdf: "[[nerf.pdf]]"
original title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
---

Note: this is not an easy-to-understand paper. I get the basic idea of it by first going over [a YouTube video](https://youtu.be/CRlN-cYFxTk?si=SBvr65BjiHjWcMyp).

## Overview

The most novel idea in this paper is that: it use a trained neural network to represent a scene. If you want to describe a new scene, you need a set of new weights for that scene. To train such a network (function), you just need some images.

![[nerf_inout.png]]

Another idea is to represent 3d surfaces by Neural Radiance Fields (that's in the title), in contrast to Signed Distance Function used in [[DeepSDF]] (recall that's converting surface samples to a "surface function"), or grid based methods.

![[nerf_overview.png]]
The core part of this image is "differentiable". Training input and inference inputs are the same, so the process is basically "try rendering a bunch of images based on network weight, and see if the image look like the real images".

What's volume density? Obviously it's classic thing in volume rendering.

> The volume density $\sigma(x)$ can be interpreted as the differential probability of a ray terminating at an infinitesimal particle at location $x$. The expected color $C(r)$ of camera ray $r(t) = o + td$ with near and far bounds $t_n$ and $t_f$ is:

$$C(\mathbf{r})=\int_{t_n}^{t_f}T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})dt\mathrm{~,~where~}T(t)=\exp\left(-\int_{t_n}^t\sigma(\mathbf{r}(s))ds\right).$$

$T(t)$ here is basically the probability that the ray travels from $t_n$ to $t$ without hitting any other particle (it's not occluded). So here the formula means "if the ray ends here you would see this, if it ends there you'll see that, add them together".

### Details

- In practice, the volume rendering is done by something called quadrature, with stratified sampling.
- Density only takes in location $x$ as input, while color also takes viewing angles into account.

## In training (optimizing):

### Positional encoding

Directly using $xyz\theta\phi$ cause the model to perform poorly. This can be caused by the networks are biased towards learning lower frequency functions. So we need to map them to higher dimension space. A position encoding similar to the [[Attention is all you need|OG Transformer]] is used. It's also multiplied with a learned encoding.

### Hierarchical volume rendering

Spend the network power on where it's needed. Train two networks: one "coarse" and one "fine". First sample a set of $N_{c}= 64$ locations, then get a PDF of where it think is dense, and then use inverse transform sampling to sample fine points $N_{f}=128$.
The loss is the simple summation of both networks.

$$\mathcal{L}=\sum_{\mathbf{r}\in\mathcal{R}}\left[\left\|\hat{C}_c(\mathbf{r})-C(\mathbf{r})\right\|_2^2+\left\|\hat{C}_f(\mathbf{r})-C(\mathbf{r})\right\|_2^2\right]$$

Here $C$ is the rendered RGB color.

Training takes about 1-2 days on a V100.
