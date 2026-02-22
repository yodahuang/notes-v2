---
date: 2026-02-04
aliases:
  - DAgger
---

Source: CS 285 at UC Berkeley
- Lecture Recording from 2023
- Lecture note from 2026: [[behavior_cloning_1.pdf]], [[behavior_cloning_2.pdf]]

Behavior cloning is supervised learning. 
### Distribution shift

Vanilla version does not work because we train on $p_{\text{train}}(x)$ and test on $p_{\text{test}}(x)$.

It can be proofed (with some nice modeling) that the error increases quadratically with horizon. This is from the DAgger paper. Why haven't Drew publish more paper recently? 

A way to fix this distribution shift is, unsurprisingly, proposed from the DAgger paper: Dataset Aggregation. 

1. Train $\pi_{\theta}(\mathbf{a}_t|\mathbf{o}_t)$ from human data $\mathcal{D} = \{\mathbf{o}_1, \mathbf{a}_1, \dots, \mathbf{o}_N, \mathbf{a}_N\}$
2. Run $\pi_{\theta}(\mathbf{a}_t|\mathbf{o}_t)$ to get dataset $\mathcal{D}_{\pi} = \{\mathbf{o}_1, \dots, \mathbf{o}_M\}$
3. Ask human to label $\mathcal{D}_{\pi}$ with actions $\mathbf{a}_t$
4. Aggregate: $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_{\pi}$

It's proved that this can reduce the error to linear. 

### Why might we fail to fit the expert
1. Non-Markovian behavior
	1. Not only depend on past single state
	2. Causal confusion (the model may mix up the cause and the result)
2. Multimodal behavior: multiple trajectories are viable, so fitting one-true one does not work
	1. Discretize continuous action space
		1. Autoregressive discretization. So an autoregressive model output one dim at a time. For sequential model, since the next output depend on all the previous output, the math works.
		2. Consider autoregressive robot output modeling. E.g. [[FAST]]
	2. Expressive continuous distributions
		1. [[Variational Autoencoder (VAE)|VAE]]
		2. [[Flow Matching]] / [[Score matching]]

### Data

Intentionally add *mistakes* and *corrections*, possibly with data augmentations. That's also where the idea of pre-training and post-training comes up, where we first use diverse knowledge and then narrow, high quality data later.

### Multi-task learning

Maybe teach the model to reach *any* p help it better to learn how to reach $p_1$.