---
aliases: [FPN]
share: true
---

[Arxiv link](https://arxiv.org/abs/1612.03144)

![[ML/papers/imgs/fpn_1.png]]

![[ML/papers/imgs/fpn_2.png]]

Similar to [[SSD]], use the structure in CNN to mimic image pyramid commonly used in traditional CV. The novel part is that it combines the deeper, but semantically richer layer with the shallower one, form a “skip connection”. It’s only tested as RPN or in fast RCNN.