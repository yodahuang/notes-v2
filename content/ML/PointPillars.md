---
Arxiv: https://arxiv.org/abs/1812.05784
pdf: "[[pointpillars.pdf]]"
original title: "PointPillars: Fast Encoders for Object Detection from Point Clouds"
date: 2025-02-07
tags: 
year: 2019
---
A classic paper that I should have read long ago. Alex H. Lang also interviewed me when I was looking for my first full time job.

The main contribution of the work is a way to represent 3D lidar points in a 2D pseudo image form, suitable for classic image backbones. The resulting network is fast and with good enough performance.

At a high level, we:
- Bin the lidar points in x-y plane into grids.
- For each points, there's location info, and we augment it with relative location info w.r.t. the pillar.
- Put all these non-empty pillars into a tensor, with max points.
- Run a (simplified) [[PointNet]] on the set of pillars, which is FC followed by max pooling.
- Scatter the features back to the original image.

The formal one is this:
> To apply a 2D convolutional architecture, we first convert the point cloud to a pseudo-image.
We denote by $l$ a point in a point cloud with coordinates $x,y,z$ and reflectance $r.$ As a first step the point cloud is discretized into an evenly spaced grid in the x-y plane, creating a set of pillars $\mathcal{P}$ with $|\mathcal{P}|=B.$ Note that there is no need for a hyper parameter to control the binning in the z dimension. The points in each pillar are then augmented with $x_c,y_c,z_c,x_p$ and $y_p$ where the $c$ subscript denotes distance to the arithmetic mean of all points in the pillar and the $p$ subscript denotes the offset from the pillar $x,y$ center. The augmented lidar point $l$ is now $D=9$ dimensional.
>
> The set of pillars will be mostly empty due to sparsity of the point cloud, and the non-empty pillars will in general have few points in them. For example, at $0.16^2m^2$ bins the point cloud from an HDL-64E Velodyne lidar has 6k-9k non-empty pillars in the range typically used in KITTI for $\sim97\%$ sparsity. This sparsity is exploited by imposing a limit both on the number of non-empty pillars per sample $(P)$ and on the number of points per pillar $(N)$ to create a dense tensor of size $(D,P,N).$ If a sample or pillar holds too much data to fit in this tensor the data is randomly sampled. Conversely, if a sample or pillar has too little data to populate the tensor, zero padding is applied.
>
> Next, we use a simplified version of PointNet where, for each point, a linear layer is applied followed by Batch- Norm and ReLU to generate a $(C, P, N)$ sized tensor. This is followed by a max operation over the channels to create an output tensor of size $(C, P)$. Note that the linear layer can be formulated as a 1x1 convolution across the tensor resulting in very efficient computation. Once encoded, the features are scattered back to the original pillar locations to create a pseudo-image of size $(C, H, W)$ where $H$ and $W$ indicate the height and width of the canvas.

[[pointpillars.pdf#page=3&selection=176,0,301,45|pointpillars, page 3]]

The encoding part is got from [[VoxelNet]] but add the offsets. The paper's experiment setup mostly follows that too.
## Details

To get good performance, these augments are applied
- Get the associated lidar points for ground truth boxes. Sample some to put into current point cloud.
- Rotate or translate the ground truth boxes.
- Global random flip along the x axis, rotation, scaling, small translation.
This is for KITTI.