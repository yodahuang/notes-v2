---
Arxiv: https://arxiv.org/abs/2106.09685
pdf: "[[lora.pdf]]"
original title: "LoRA: Low-Rank Adaptation of Large Language Models"
date: 2024-11-30
tags: 
year: 2021
---
Model fine tuning: fast, memory efficient and good enough.

Why it's good:
- We can swap out the fine-tuning LoRA models for different downstream tasks.
- Trains fast. No big memory needed.
- No inference latency.

> For a pre-trained weight matrix $W_0\in\mathbb{R}^{d\times k}$, we constrain its update by representing the latter with a low-rank decomposition $W_0+\Delta W=W_0+BA$,where $B\in\mathbb{R}^d\times r,A\in\mathbb{R}^{r\times\tilde{k}}$, and the rank $r\ll\min(d,k).$ During training, $W_0$ is frozen and does not receive gradient updates, while $A$ and $B$ contain trainable parameters. Note both $W_0$ and $\Delta W=BA$ are multiplied with the same input, and their respective output vectors are summed coordinate-wise. For $h=W_0x$, our modified forward pass yields:
>
> $$
> h=W_0x+\Delta Wx=W_0x+BAx
> $$
>
> We illustrate our reparametrization in Figure 1. We use a random Gaussian initialization for $A$ and zero for $B$, so $\Delta W=BA$ is zero at the beginning of training. We then scale $\Delta Wx$ by $\frac\alpha r$, where $\alpha$ is a constant in $r$. When optimizing with Adam, tuning $\alpha$ is roughly the same as tuning the learning rate if we scale the initialization appropriately.  

[[lora.pdf#page=4&selection=113,84,272,49|lora, page 4]]

Now, applying this to Transformer. 

> We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity and parameter-efficiency

[[lora.pdf#page=5&selection=54,22,60,23|lora, page 5]]

> Note that putting all the parameters in $\Delta W_q$ or $\Delta W_k$ results in significantly lower performance, while adapting both $W_q$ and $W_v$ yields the best result. This suggests that even a rank of four captures enough information in $\Delta W$ such that it is preferable to adapt more weight matrices than adapting a single type of weights with a larger rank.

[[lora.pdf#page=10&selection=194,0,226,53|lora, page 10]]

Well, $k$ or $q$ shouldn't really matter here.

Lower rank of LoRA, even 1 or 2 is pretty good, this can be further understood by looking at the subspace similarity, the top values are the most useful, other directions are likely random noise.

> First, $\Delta W$ has a stronger correlation with $W$ compared to a random matrix, indicating that ∆W amplifies some features that are already in $W$ . Second, instead of repeating the top singular directions of $W$ , $\Delta W$ only amplifies directions that are not emphasized in $W$ . Third, the amplification factor is rather huge.

[[lora.pdf#page=12&selection=343,14,381,45|lora, page 12]]