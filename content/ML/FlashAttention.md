---
aliases: 
Arxiv: https://arxiv.org/pdf/2205.14135
pdf: "[[flash_attention.pdf]]"
original title: "FLASHATTENTION: Fast and Memory-Efficient Exact Attention with IO-Awareness"
date: 2024-12-03
tags:
---
Flash Attention speed up attention computation by
- Incrementally compute softmax without fully materialize the full matmul result. (tiling)
- Recompute intermediate result instead of storing them in backward pass by storing some extra info.
![[flash_attention.png]]
What’s not covered in the paper is “how do you know it’s HBM access making it slow” in the first place.

### Standard attention
$N$ is sequence length and $d$ is the head dimension

$$
kS = QK^{T} \in \mathbb{R}^{N\times N}, \ \ P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \ \ O = PV \in \mathbb{R}^{N \times d}
$$

![[regular_attention.png]]

### Flash Attention
TODO: Add the Latex formula here. 
> The backward pass typically requires the matrices $S, P \in \mathbb{R}^{N \times N}$ to compute the gradients with respect to Q, K, V. However, by storing the output O and the softmax normalization statistics (𝑚, ℓ), we can recompute the attention matrix S and P easily in the backward pass from blocks of Q, K, V in SRAM. 

[[flash_attention.pdf#page=5&selection=305,43,359,9|flash_attention, page 5]]

![[flash_attention_algo.jpg]]
