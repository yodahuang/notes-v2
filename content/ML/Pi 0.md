---
aliases:
Arxiv: https://arxiv.org/abs/2410.24164
pdf: "[[pi0.pdf]]"
original title: "π0: A Vision-Language-Action Flow Model for General Robot Control"
date: 2025-12-20
tags:
year: 2024
---
The paper mainly focus on the dataset / experiment setup, and show that "it can work" as a system, toward generalist robot.

End result: a model that takes in images and pose, and can one shot easy tasks. Complex, longer horizon tasks needs fine tuning.

![[pi0_model.png]]

The model structure is heavily inspired by [[TransFusion]] for supervising continuous output with flow matching objective. Unlike that paper, a separate set of weights is used for robotics-specific (action and state) tokens.

As is shown in the image, they used a pre-trained VLM, [[PaliGemma]]. We then added an action expert, but note, it's still on transformer. 

> Attention mask. π0 uses a blockwise causal attention mask with 3 blocks: $[I^{1}_{t}, \dots I^{n}_{t}, l_t]$, $[q_t]$, and $[a^\tau_t , \dots, a^\tau_{t+H-1}]$. Within each block, there is full bidirectional attention, whereas the tokens in each block cannot attend to the tokens in future blocks. The first block includes the input modalities from PaliGemma’s VLM pre-training, which are prevented from attending to future blocks (which include new inputs) to minimize distribution shift from said pre-training. The robot state qt is its own block because it does not change with each flow matching integration step; preventing it from attending to the final block allows its corresponding keys and values to be cached during sampling. The final block corresponds to the noisy actions $A^\tau_t$ , which can attend to the full input sequence.

[[pi0.pdf#page=15&selection=408,0,476,46|pi0, page 15]]

> We sample τ from a shifted beta distribution that emphasizes lower timesteps (corresponding to noisier actions), and does not sample timesteps at all above a cutoff value s. We use s = 0.999 in our experiments.

[[pi0.pdf#page=16&selection=17,0,34,19|pi0, page 16]]

The model output is a "trajectory", an action sequence of length 50, where each element is an action chunk. It's unclear how the "chunk" is being defined. The loss is a conditional flow matching loss with a simple linear gaussian (or optimal transport) probability path.

In inference time, 10 integration steps are used with KV cache. Inference is open loop. Ensemble hurts performance. 

### Dataset

> Intuitively, the diverse (but lower quality) pre-training data allows the model to recover from mistakes and handle highly varied situations, which might not otherwise occur in the high-quality post-training data, while the post-training data teaches the model to perform the task well.

[[pi0.pdf#page=5&selection=518,10,522,43|pi0, page 5]]

Only 9.1% is from open-source datasets, while the rests are from their own datasets from various different robots of 68 tasks. The class imbalance is offset by weighing by $n^{0.43}$. How suspicious that number is. They also use a subset from OXE, called "OXE Magic Soup". Sus. 

The experiment result shows that with pretraining, the model performs even better than the from-scratch version with fine tuning data. Using large VLM may be helpful, but it's hard to tell if it's because they use web scale data, or just because the model is large.


# Note to self

It's unclear to me how does the "mixture of expert" action expert actually work. Routing specific token to specific expert? Maybe I should read the MoE paper.