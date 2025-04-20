---
aliases: 
Arxiv: https://arxiv.org/abs/2301.08739
pdf: "[[flatformer.pdf]]"
original title: "FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer"
date: 2025-04-04
tags: 
year: 2023
---
Makes [[SST (Single Stride 3D Detector)|SST]] faster. If they are from the same lab, the name would just be FastSST.

Note they claim they are 5 times faster than [[SST (Single Stride 3D Detector)|SST]] , don't trust that. They count in speed boost by FlashAttention and Triton kernel in FFN. 

Some background: why were point cloud transformers slow? Global attention is of course slow. Local attention suffer from neighbor preparation overhead. Window attention have different number of points in a window so they need to do batch padding.

Well wait, what if we fore them to have the same number of points in the window? That's basically this paper.
![[flatformer.png]]
Let's just make a way of having an equal number of points per group, and make the group preserve locality as much as possible. 

With a point cloud $\{(x,y)\}$,we first quantize the coordinate of each point $(x,y)$ to

$$\left(\underbrace{\lfloor x/w_x\rfloor,\lfloor y/w_y\rfloor}_{\text{window coordinates}},\underbrace{x-\lfloor x/w_x\rfloor\cdot w_x,y-\lfloor y/w_y\rfloor\cdot w_y}_{\text{local coordinates within window}}\right),$$

where $(w_x,w_y)$ is the window shape. Next, we sort all points first by window coordinates and then by local coordinates within the window. This step turns the unordered point cloud into an ordered one, where points within the same window will be next to each other.
[[flatformer.pdf#page=4&selection=244,0,341,27|flatformer, page 4]]

Then we just do equal size grouping. The image shows a "good case". You can also imagine the $D$ is around the left boundary. In that case it's not close to $I$ and $F$ at all. Surprisingly this doesn't matter much to the network.

> There is a trade-off: equal-window grouping maintains perfect spatial proximity (i.e., each group has the same radius) but breaks the computational regularity, while equal-size grouping ensures balanced computation workload (i.e., each group has the same number of points) but cannot guarantee the geometric locality. We show in Section 5.3 that computation regularity is more important since spatial irregularity can be partially addressed by our algorithm design: i.e., window-based sorting offers a fairly good spatial ordering, and self-attention is robust to outliers (i.e., distant point pairs).

[[flatformer.pdf#page=4&selection=417,37,442,24|flatformer, page 4]]

In different attention block, we alternate between sort by $x$ or $y$ axis first, just like how in other works we decompose $3\times3$ convolution to $3\times1$ and $1\times3$ blocks.

There's also [[Swin Transformer]] style window shift, where the window got shifted.y

> Interestingly, despite the fact that our sorting strategy does not guarantee the windows to be geometrically regular as in SST, FlatFormer still consistently outperforms SST in all three classes.

[[flatformer.pdf#page=7&selection=233,0,236,18|flatformer, page 7]]. So the author doesn't understand why either. There's also an ablation study shows using group size with $85\%$ of window size provide the better result.