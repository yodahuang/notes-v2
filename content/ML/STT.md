---
original title: "STT: Stateful Tracking with Transformers for Autonomous Driving"
date: 2024-05-10
pdf: "[[stt.pdf]]"
---
Waymo folks' paper are always hand wavy and this one is no exception. This one is not from Dragomir though.

The paper covers both a model and some metrics. These metrics are not interesting. Not gonna cover it here.

![[stt_overview.png]]

The task here is: given good detector, how to do good tracking. STT proposes we jointly optimize association and state estimation, with Transformers all over the place. Take a look at the image above:
- Detection Encoder is a MLP
- Temporal Fusion means they concat encoded detection together, self attention, then condense it to be an embedding (the track query)
- Context detections means detections geometrically around
- Track-Detection interaction Module: a transformer, with both track query and detections as input. How exactly? Not mentioned. My guess is it's a decoder only model, where it attends to track query, while using detections as query, so it can output one score per detection.

Online inference is Hungarian matching. What about training? That's where the hand wavy kicks in. It's not mentioned. Importantly, where does the "track with detection history" comes from?
- It can be coming from this exact model with online inference. That'd be slow since it cannot be parallelized. Training can also be unstable in the beginning
- Or some other way?

