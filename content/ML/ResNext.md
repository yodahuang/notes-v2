---
aliases: 
Arxiv: https://arxiv.org/abs/1611.05431
pdf: "[[resnext.pdf]]"
original title: Aggregated Residual Transformations for Deep Neural Networks
date: 2024-10-26
tags: 
year: 2017
---

One and half a year later, again from [[Kaiming]]. 
![[resnext_block.png]]
![[resnext_equivalent_blocks.png]]
Here the team takes a note at [[Inception]]'s slit-transform-merge strategy, and combine it with [[VGG]] or [[ResNet]]'s simple structure. By doing so, with same FLOPs and parameter count, we got better performance. Note that this doesn't mean network runs at the same speed though.

> Our method indicates that cardinality (the size of the set of transformations) is a concrete, measurable dimen- sion that is of central importance, in addition to the dimen- sions of width and depth. Experiments demonstrate that in- creasing cardinality is a more effective way of gaining accu- racy than going deeper or wider, especially when depth and width starts to give diminishing returns for existing models.

[[resnext.pdf#page=2&selection=51,0,64,61|resnext, page 2]]

Note there's something in the paper that I don't understand:

> We note that the reformulations produce nontrivial topologies only when the block has depth ≥3. If the block has depth = 2 (e.g., the basic block in [14]), the reformula- tions lead to trivially a wide, dense module. See the illus- tration in Fig. 4.

[[resnext.pdf#page=4&selection=297,0,322,18|resnext, page 4]]

In the Experiments section, they explained that increasing cardinality (more groups) is better than going deeper or wider.