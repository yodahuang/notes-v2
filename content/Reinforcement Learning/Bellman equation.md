---
date: 2025-10-19
---

$$
\begin{align*}
v_{\pi}(s) &= \mathbb{E} \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t \sim \pi(s) \right] \\
&= \mathbb{E} \left[ R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t \sim \pi(s) \right]
\end{align*}
$$

Now note here that $a$ is chosen by policy $\pi$ in state $s$.
We can take out the dependency on a specific policy by stating that the following holds for the optimal case: 

$$v_*(s)=\max_a\mathbb{E}\left[R_{t+1}+\gamma v_*(S_{t+1})\mid S_t=s,A_t=a\right]$$

For Bellman equation, in discrete case we can write it out in linear form. In fact we can just solve it given $\pi$, the same way we solve MDPs by finding the stationary point.

$$\mathbf{v} = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}$$

where

$$
\begin{aligned}
v_i &= v(s_i) \\
r_i^\pi &= \mathbb{E}[R_{t+1} | S_t = s_i, A_t \sim \pi(S_t)] \\
P_{ij}^\pi &= p(s_j | s_i) = \sum_a \pi(a | s_i) p(s_j | s_i, a)
\end{aligned}
$$

Solving it is $O(|S|^3)$ though.
The Bellman optimality equation though, is non-linear, there's the max there.