---
aliases:
Arxiv: https://arxiv.org/abs/2405.15613
pdf: "[[hierarchical-k-means.pdf]]"
original title: "Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach"
date: 2025-11-30
tags:
year: 2024
---
We want to curate data. More specifically, for [[self supervised learning]]. 

Here's the poorly plotted overview.
![[hierarchal k-means.png]]

First, it seems like a good dataset not only should be large and divers, it should also be balanced. This is shown by training on artificially unbalanced ImageNet. 
One way to do that is to assign "concept" to each sample and uniformly sample that. We can approximate that by clustering based on embeddings (In this paper they use [[DINOv2]]). Do notice that a sample may have multiple concepts, and this cluster approach seems like an approximation.
![[why_kmeans_is_not_enough.png]]
![[hierarchical-may-be-better.png]]
It can be proved that with the application of K-means, the new distribution is closer to the uniform data representation (where it is defined on space with data support) compared with the original distribution.
![[more-hierarchical-core-idea.png]]

But after each layer we have exponentially less points.
![[hierarchical-kmeans-resampling-clusetering.png]]
![[hierrarchical-kmeans-sampling.png]]

![[hierarchical_kmeans_ablation.png]]