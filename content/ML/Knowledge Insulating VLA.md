---
aliases:
Arxiv:
pdf: "[[pi05_KI.pdf]]"
original title: "Knowledge Insulating Vision-Language-ActionModels: Train Fast, Run Fast, Generalize Better"
date: 2026-01-18
tags:
year: 2025
---
![[pi_0_5_ki_comparison.png]]

This paper basically eliminate that two stage training in [[Pi 0.5]]. Remember we need to add action expert / switch loss target in the training process? With this thing you don't need to do that anymore. 

Naively co-training both would lead to worse performance:

> prior approaches for finetuning VLMs with continuous outputs can, perhaps unsurprisingly, lead to significantly worse training dynamics, as they rely on gradients from continuous adapters (e.g. diffusion heads) for the training signal. This can degrade both their ability to interpret language commands and the overall performance of the resulting VLA policy.

[[pi05_KI.pdf#page=2&selection=134,30,137,95|pi05_KI, page 2]]

And freezing the VLM is also a good idea: 

> However, current VLMs are not pre-trained with robotics data. As a result, their representations, when frozen, are insufficient for training highly performant policies, as we show in our experiments

[[pi05_KI.pdf#page=5&selection=95,9,99,57|pi05_KI, page 5]]

### Solution

> Therefore, we propose to stop the gradient flow from the action expert to the pre-trained weights in the model. This is a sensible restriction **if and only if** the backbone is additionally trained to predict actions directly as part of its language outputs. 

[[pi05_KI.pdf#page=6&selection=189,70,192,30|pi05_KI, page 6]]

The new loss:

$$\mathcal{L}_{\text{CO-VLA}}(\theta) = \mathbb{E}_{\mathcal{D},\tau,\omega} \left[ -\sum_{j=1}^{n-1} M_j^{\ell} \log p_{\theta}(\hat{\ell}_{j+1}|x_{1:j}) + \alpha M^{\text{act}} \|\omega - a_{1:H} - f_{\theta}^a(a_{1:H}^{\tau,\omega})\|^2 \right]$$

where $\alpha$ is a loss multiplier, trading off action prediction via flow-matching with the standard language modeling loss. $M^{\ell}$ is a language loss mask (indicating locations in the token stream at which the language loss should be applied) and $M^{\text{act}}$ is an action mask indicator specifying whether or not actions should be predicted for the given example.

Compare with [[Pi 0.5]], we can see the main difference is that mask. 

> This loss construction allows us to flexibly mix-and-match co-training with data from different modalities. In particular, we combine *VLM* data (which has only images and text annotations) with *action-only* data (where the task is action prediction conditioned on images and text) as well as *combined language and action prediction* tasks (where we take action only data and additionally annotate it with a language description of what the robot should do next)

[[pi05_KI.pdf#page=6&selection=147,0,163,30|pi05_KI, page 6]]

[[Pi 0.5]] has this two stage training that in the second stage, it weights by setting $\alpha = 10$ and train both action expert and the VLA backbone. This value is set to focus more on the action expert and avoid it corrupting the VLA too much. Here we use a better method, so we can still let VLA adapt to robot data, without corrupting it. (If we freeze it after pretraining, it learns less)

For the single head attention case, we can write the attention operation as $P = \text{softmax}(Q(X)K(X)^T + A) = \begin{pmatrix} P_{bb} & 0 \\ P_{ab} & P_{aa} \end{pmatrix}$ where $X$ are the inputs to the attention layer, $Q, K$ are the attention query and key projections, respectively, $A$ is the attention mask as described above, and softmax is the row-wise softmax. The result are attention probabilities over token features which decompose into probabilities where features from the VLM backbone attend to features from the backbone $P_{bb}$, probabilities for action expert features attending to backbone features $P_{ab}$ and probabilities for action expert features attending other action expert features $P_{aa}$. Given this we can restrict information flow as desired by implementing the softmax computation as

$$\begin{pmatrix} P_{bb} & 0 \\ P_{ab} & P_{aa} \end{pmatrix} = \text{softmax} \left( \begin{pmatrix} Q_b(X_b)K_b(X_b)^T & 0 \\ Q_a(X_a)\text{sg}(K_b(X_b)^T) & Q_a(X_a)K_a(X_a)^T \end{pmatrix} + A \right)$$

where sg denotes the stop-gradient operator that restricts gradient-flow through this part of the computation. $X_b$ corresponds to all $x_i$ processed with the backbone weights, $X_a$ to the tokens processed with the action expert weights. The value embeddings are then computed by

$$E = \begin{pmatrix} E_b \\ E_a \end{pmatrix} = \begin{pmatrix} P_{bb}V_b(X_b) \\ P_{ab}\text{sg}(V_b(X_b)) + P_{aa}V_a(X_a) \end{pmatrix}$$

and the final attention is $\text{attn}(X) = PE$. One additional advantage of this design is that we can simply set $\alpha = 1$ in (4), since now the diffusion loss term applies to an independent set of weights.

In other words, as we cut the gradient on backbone features used in key calculation for action expert attending, action expert's loss term would not affect backbone (kinda). The action loss can still update backbone parameters, but stop-gradient prevents updates flowing _through the action→backbone cross-attention K/V pathway_.