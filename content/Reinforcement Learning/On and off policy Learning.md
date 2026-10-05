---
date: 2025-11-02
updated: 2026-10-04
aliases:
  - Off-Policy
  - On-Policy
---
### **On-policy** learning
- Learn about **behaviour** policy $\pi$ from experience sampled from $\pi$

### **Off-policy** learning
- Learn about **target** policy $\pi$ from experience sampled from $\mu$
- Learn ‘counterfactually’ about other things you could do: “what if...?”
  - E.g., “What if I would turn left?” $\implies$ new observations, rewards?
  - E.g., “What if I would play more defensively?” $\implies$ different win probability?

Evaluate target policy $\pi(a \mid s)$ to compute $v_\pi(s)$ or $q_\pi(s, a)$  
While using behaviour policy $\mu(a \mid s)$ to generate actions  
#### Why is this important?
- Learn from observing humans or other agents (e.g., from logged data)  
- Re-use experience from old policies (e.g., from your own past experience)  
- Learn about **multiple** policies while following **one** policy  
- Learn about **greedy** policy while following **exploratory** policy  

## Why one-step TD works off-policy

This is the argument behind off-policy [[Q learning]], the replay buffer in [[Actor-Critic#Off-policy actor-critic]], and [[Fitted Q Iteration]].

A policy decides **which** $(s, a)$ pairs you visit. It does not decide what the environment does once $s, a$ are fixed. Writing out where a replayed transition comes from:

$$
p_\mu(s, a, r, s') = \underbrace{d_\mu(s)}_{\text{states } \mu \text{ reaches}} \; \underbrace{\mu(a \mid s)}_{\text{actions } \mu \text{ picks}} \; \underbrace{P(r, s' \mid s, a)}_{\text{environment}}
$$

Condition on the $(s, a)$ you sampled and the first two factors drop out: $p_\mu(r, s' \mid s, a) = P(r, s' \mid s, a)$. So $(r, s')$ is a genuine sample of the environment's response to that action, whether a random policy, a human, or last week's network chose it. The Bellman target

$$
y = r + \gamma\, Q(s', a'), \qquad a' \sim \pi(\cdot \mid s')
$$

then has the right expectation for $Q^\pi(s, a)$. The transition comes from the buffer, and the *next* action is asked fresh from the current $\pi$ (or the $\max$ for Q-learning).

> [!warning] Coverage is the catch
> The behaviour policy doesn't bias the target, but it decides what you get to see. If $\mu$ never takes $(s^\star, a^\star)$, the data says nothing about $P(\cdot \mid s^\star, a^\star)$, and any $Q$ value there is pure extrapolation by the network. Offline RL lives and dies by this: $Q$ can be confidently wrong on unseen actions, and a policy that maximizes $Q$ goes looking for exactly those errors. Hence behaviour constraints, conservatism, BC regularization, etc.

## Why n-step returns break this

At one step, nothing chosen by $\mu$ sits between the action we condition on and the bootstrap. At two steps, one does:

$$
r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}), \qquad r_{t+1} \sim P(\cdot \mid s_{t+1}, \underbrace{a_{t+1}}_{\sim\, \mu,\ \text{not } \pi})
$$

$Q^\pi(s_t, a_t)$ means "take $a_t$, *then follow $\pi$*", but the replayed $r_{t+1}, r_{t+2}, \dots$ came from following $\mu$. An $n$-step return has $n-1$ of these unconditioned behaviour actions, which is why it's only correct on-policy (see [[Q learning#N step returns]]).

Two ways out:
- Reweight by $\prod \pi / \mu$: [[Importance sampling corrections]].
- Change the question so those actions are conditioned on too: $Q(s_t, a_t, \dots, a_{t+n-1})$ asks "what if I execute *exactly this sequence*?", and the replayed rewards answer it with no bias. That's [[Q-chunking]].
