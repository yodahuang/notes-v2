---
date: 2025-10-26
---

Model free unbiased prediction method.

Recall the value function (given a policy) by definition is 
$$\nu_\pi(s)=\mathbb{E}\left[G_t\mid S_t=s,\pi\right]$$
$$
G_t=R_{t+1}+\gamma R_{t+2}+...+\gamma^{T-t-1}R_T
$$
We can just replace that expectation with sampling average to get value estimation.