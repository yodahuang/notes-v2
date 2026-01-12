---
aliases:
  - LSS
Arxiv: 
pdf: "[[lift_splat_shoot.pdf]]"
original title: ": Encoding Images fromArbitrary Camera Rigs by ImplicitlyUnprojecting to 3D"
date: 2025-12-31
tags: 
year: 2020
---
The paper focus on detetion and planning based on multiple cameras.

> A potential performance bottleneck of these models that our model addresses is that a pixel contributes the same feature to every voxel independent of the depth of the object at that pixel.

[[lift_splat_shoot.pdf#page=4&selection=43,15,45,41|lift_splat_shoot, page 4]]

That's why this method is better than pseudo-lidar.

## Lift

For a $3 \times H \times W$ image, we associate $|D|$ points to each pixel where $D$ is a set of discrete paths. For each pixel, we predicts a context (as normal) and a distribution over depth. For each point in that point cloud, we then attach the context of the pixel times the probability of depth. 

## Splat

Now we convert things to BEV with [[PointPillars]]. Each point is assigned to a pillar and we do max pooling. 

Different from PointPillars, they cannot use PointNet for exploiting the sparsity of lidar points. After all it's generated 3D grid. But due to projection, the BEV grid could still have dense and sparse parts. Now we need padding and that cost excessive memory. 

Now think: how can we do streaming max pooling? The author argues that we need to use sum pooling because that's the more memory efficient way and I don't get that. They argue we can use cumsum trick to get the individual bin sum.

## Shoot

Not interested for me.