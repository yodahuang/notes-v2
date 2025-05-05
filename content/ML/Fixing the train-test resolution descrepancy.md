---
aliases: 
Arxiv: https://arxiv.org/abs/1906.06423
pdf: "[[fixing-train-test.pdf]]"
date: 2025-05-04
tags: 
year: 2019
---
TL;DR: If you use `ResizedRandomCrop`, then in training time you'll likely see a zoomed in image. To counter that effect, you need to make your test time image and crops larger, and fine tune the last layers, especially BatchNorm.

![[fixing_train_test.png]]

How to read the above image:
- The "standard pre-processing" here means `ResizedRandomCrop` for train and `CenterCrop` for test. 
- As a result you can see the horse is super small in test.
- Now our goal is to make the horse to have the same pixel size. We can either align with test, or align with train size. The crop size of for test would be larger, but that's fine as long as the apparent size (pixel size) stays the same.

Now onto the details.

Imagine a classic pinhole camera model. For most of the cameras, their field of view angle stays within a small range. That means the "mm" size would be roughly the same. What's different is the resolution, or the sampling rate. Our network only care about the pixel size.

At training time:
$$
r_{train} = \frac{kK_{train}}{\sigma}\cdot r_{1}
$$
- $r_{train}$: apparent size of training
- $k$: variable relate to focal length. $k^{-1}\approx 1$
- $\sigma$: scale parameter for `SizedRandomCrop`
- $r_1$: $\frac{R}{Z}$, where $R$ is size and $Z$ the depth. So only related to the object

At test time, usually we isotropically resizing the image so that the shorter dimension is $K_\mathrm{test}^\mathrm{image}$ and then extracting a $K_\mathrm{test}\times K_\mathrm{test}$ crop (CenterCrop) from that.
$$r_{test} = kK^{image}_{test}\cdot r_1$$

Thus, we should increase $K^{image}_{test}$ by $\frac{1}{\alpha}$ too to counter that. That can be intuitively understood as: in training augmentation we zoom in. So test time we should "zoom in" too by sampling more. We also increase $K_{test}$ to keep the crop / image ratio be the same, so we are not looking at nothing.

Now $K_{test}$ is different from $K_{train}$, and that causes a problem since the pooling layers (note the pool kernel size is fixed) or BatchNorm may not lit it. (Note we assume the network's last layer is maxpool, not FC, so it can handle images of different sizes). So we're gonna fine-tune the last layers.

