---
date: 2026-02-22
---
An algorithm proposed by R. Sutton.

1. collect some data, consisting of transitions $(s, a, s', r)$
2. learn model $\hat{p}(s'|s, a)$ (and optionally, $\hat{r}(s, a)$)
3. repeat $K$ times:
    1. sample $s \sim \mathcal{B}$ from buffer
    2. choose action $a$ (from $\mathcal{B}$, from $\pi$, or random)
    3. simulate $s' \sim \hat{p}(s'|s, a)$ (and $r = \hat{r}(s, a)$)
    4. train on $(s, a, s', r)$ with model-free RL
    5. (optional) take $N$ more model-based steps

![[Dyna.png]]
The algorithm mentioned here is close to MBPO.

![[model_accelerated_off_policy_rl.png]]