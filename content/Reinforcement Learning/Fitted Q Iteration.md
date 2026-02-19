---
date: 2026-02-11
pdf: "[[285-value.pdf]]"
---
Coming from [[Fitted Value Iteration]], we don't have transition dynamics. $Q$ is good (we can see the same idea in [[Actor-Critic#Off-policy actor-critic]]) as the reward now does not depend on your policy, just $r(s, a)$, not $r(s, \pi(s))$.

Recall previously we had ![[fitted_value_iteration.png]] Now it's ![[fitted_q_iteration.png]]

Note how the $\max_{a_i}$ disappears, so this works even for off-policy samples.

**Why fitted value iteration needs the environment but fitted Q doesn't:** In fitted value iteration, we compute $\max_a [r(s_i, a) + \gamma \mathbb{E}[V(s')]]$. Our dataset only has one tuple $(s_i, a_i, r_i, s'_i)$ — we took one action and saw one outcome. To evaluate the other actions inside that max, we'd need to know what reward and next state each action leads to, which is exactly the transition model $P(s'|s_i, a)$. These are two sides of the same coin: we're missing data for actions we didn't take, _because_ we lack transition knowledge.

In fitted Q iteration, there's no max over actions at $s_i$. We just use our one tuple as-is: $y_i = r_i + \gamma \max_{a'} Q_\theta(s'_i, a')$. The max happens at $s'_i$ and only requires forward passes through our own network $Q_\theta$ — no environment interaction, no missing data.

> [!tip] The $\max_{a'} Q_\theta(s'_i, a')$ is "free"
> This max only queries our own function approximator at different action inputs. Compare with the max in fitted value iteration, where the quantity inside the max ($r(s,a) + \gamma \mathbb{E}[V(s')]$) depends on the environment's dynamics. That's the fundamental asymmetry.

This is off policy, and the $\pi$ only show up implicitly in that $\max_{a_i'}$, since we know it's max by asking the policy to estimate $Q$ at $s_i'$. (Our policy is greedy, so policy and Q are basically the same).

![[fully_fitted_q.png]]

We can make it online:

![[online_q_iteration.png]]
And for exploring in step one we can use [[Multi-arm bandits#Greedy and epsilon greedy]]. You can see more exploration way there too.o

Also see the traditional [[Q learning]]. The change are basically if we use batch / function approximation. 

Now we have correlated samples, on to the [[Deep Q]].