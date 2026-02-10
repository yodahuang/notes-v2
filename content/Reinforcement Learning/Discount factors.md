---
date: 2026-02-08
pdf: "[[285-actor-critic.pdf]]"
---
Source: CS285, 2023, Lecture 6.

This is normally briefly covered in [[The real overview of basics]], but it's worth considering why do we actually introduce it.

Without the discount factor, for infinite length episode, $V$ can get infinitely large. One simple *trick* is to say better get reward sooner than later, and introduce the $\gamma$, which really is changing the MDP.

$\gamma$ changes the MDP:
![[discount_factor_mdp.png]]

Note that how we handle the $\gamma$ does makes a difference. See the paper [Bias in Natural Actor-Critic Algorithms](https://people.cs.umass.edu/~pthomas/papers/Thomas2014b.pdf) for more. 

Consider how do we add $\gamma$ for [[Policy Gradient]] with causality optimization.

option 1:

$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t}) \left( \sum_{t'=t}^{T} \gamma^{t'-t} r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right)
$$

option 2:

$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t}) \right) \left( \sum_{t=1}^{T} \gamma^{t-1} r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right)
$$
If you work out the math and expand it, option 2 is the more correct one since it also penalize previous decisions, not only reward. In practice, we use option 1 though, since we want our policy to run infinitely long and we want average reward really, so option 1 may be a be a better choice. 