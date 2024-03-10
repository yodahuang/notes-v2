---
Created: 2021-10-24T15:40
share: true
---
A follow up on [[PointNet|PointNet]].

### Summary

PointNet does not catpture local structures. So we just run PointNet on small regions as feature extractor. We sample some centroids , find the points belong to it, and run PointNet. It's just like CNN for points. Each layer have a larger perception field.

### Details

![[pointnet++_1.png|pointnet++_1.png]]

For feature extraction, it's layers of sampling & grouping + pointnet, where sampling means using iterative farthest point sampling to choose a subset of points as centroids. Grouping is done using a ball query.

When we want to do per point segmentation though, we would want the output to have the same shape as input. So we do this kinda like U-net thing. Use nearest neighbor to interpolate, and add skip connection from previous layers.

![[pointnet++_2.png|pointnet++_2.png]]

For battle non-uniform point density, the author proposed 2 methods. First is using different scales. Second is to use set abstraction level on previous level + the normal one.