---
date: 2026-02-19
---
Based on a conversation with ChatGPT 5.2.

---
# Model-Based vs Model-Free RL

## Core distinction

The difference is about **access to a model**, not about tabular vs neural, or Monte Carlo vs TD.

A method is **model-based** if it can generate hypothetical transitions:

$$
s' \sim P(\cdot \mid s, a)  
$$

A method is **model-free** if it can only learn from experienced transitions:

$$
(s, a, r, s')
$$

---

## Model-Free RL

- No access to $P(s' \mid s,a)$
- Cannot simulate alternative futures
- Learns only from real interaction or logged data
- Cannot branch from arbitrary states

Examples:
- [[Q learning]]
- [[SARSA]]
- [[Policy Gradient]]
- [[Proximal Policy Optimization|PPO]]

Key property: works with a fixed dataset (offline RL).

---
## Model-Based RL

- Has access to a generative model or simulator
- Can simulate transitions from arbitrary states
- Can evaluate hypothetical action sequences
- Can branch and build search trees

Examples:
- [[Policy & value iteration|Value Iteration]] (with known model)
- MCTS
- MPC (with known or learned dynamics)
- Dyna-style methods

Key property: requires ability to generate new transitions.

---

## Planning vs Learning (orthogonal)

- Planning: uses a model to choose actions
- Learning: updates parameters from data

Examples:

- MCTS → model-based planning
- Q-learning → model-free learning
- Dyna → model-based learning + planning

---
## Litmus Test

If given only a static dataset (no simulator):
- Still works → model-free
- Breaks → model-based

---

In a deterministic case, given the model, one can actually just solve for the best action given any time horizon because it's basically an optimization problem. If it's a stochastic environment, doing those kinds of things in an open loop is a bad idea, obviously, because things could go wrong and the actions you choose are not optimal anymore. You'll need closed-loop planning.