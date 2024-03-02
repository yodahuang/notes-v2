---
share: true
---
SoDA: Multi-Object Tracking with Soft Data Association

[Arxiv Link](https://arxiv.org/abs/2008.07725)
I don't think Waymo folks know how to write a good, easy to read paper.

Core idea:
- Represent track state by aggregating detection embeddings over time. The aggregation is done by Transformer.
- Find the optimal association by comparing this track state with detection embeddings, cross entropy loss.
- This is soft data association because the network not only take the "previously confirmed" hard associated ones, it also takes into account the ones what were considered not associated.
![[soda_overview.png | 500]]
The output of the network is all the tracks, but we only have loss on the alive ones. There's also this $z_{acc}$ stuff here. It's just a ghost occlusion state. If the network choose to attend that, it means it's occluded. Note that there's only encoding here, but not decoding, cause what we want is to tell "what does this thing attend to".
I can't say I fully grasp the paper though.