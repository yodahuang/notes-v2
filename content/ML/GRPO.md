---
date: 2026-04-21
original title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
year: 2024
pdf: "[[deepseek_math.pdf]]"
---
From my conversation with Claude Sonnet 4.6

---

Essentially [[Policy Gradient|REINFORCE]] with a mean baseline — the only twist is that the baseline is estimated on-the-fly via [[Monte Carlo]] from $G$ rollouts of the _same input_, rather than from a separately trained critic. From the [[DeepSeek Math]] paper.

For a given input $q$, sample $G$ responses ${o_1, \ldots, o_G} \sim \pi_\theta$, get rewards ${r_1, \ldots, r_G}$ from a reward model, then define the advantage for response $o_i$ as:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}
$$

Then optimize with a PPO-style clipped surrogate objective using these advantages.

The group mean is an unbiased estimator of $V(q) = \mathbb{E}_{o \sim \pi}[r(q,o)]$ — a stochastic baseline that changes every rollout, as opposed to a fixed learned $V_\phi$. Each $\hat{A}_i$ depends on all other $r_j$ in the group, introducing a small self-correlation bias that is negligible for large $G$.

> [!tip] Why it works for reasoning 
> When reward is binary (correct/incorrect), within-group normalization captures _relative_ quality among responses to the same problem. A problem where the model always succeeds or always fails contributes zero gradient (std collapses) — which is correct behavior.

Both [[PPO]] and [[GRPO]] use a **learned neural reward model** (not rule-based), in the DeepSeek Math paper, which is the weak link — see [[Goodhart's Law]] and [[Process Reward Models]] for the open problems there.

## Dr. GRPO: Fixes to the Original Formulation

The original GRPO loss averages over tokens per response:

$$
\mathcal{L} = \mathbb{E}\left[\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \hat{A}_i \cdot \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]
$$

Dr. GRPO identifies two problems and removes both:

**1. Remove $1/|o_i|$** — length normalization biases toward shorter responses: same reward, shorter response → larger per-token gradient. Bad for chain-of-thought. Inherited from LM training where token-averaging makes sense; wrong semantics in the RL setting. Fix: sum token log-probs per response, normalize at the _response_ level via group advantage.

**2. Remove std from advantage** — dividing by std distorts relative magnitudes when within-group reward variance is low. If all responses are similarly good or bad, std $\approx 0$ causes instability or over-amplification of noise. Mean subtraction alone is sufficient for variance reduction:

$$
\hat{A}_i = r_i - \text{mean}(r_1, \ldots, r_G)
$$

## Connection to MCTS and PRMs

GRPO's "group rollouts from the same state" is structurally the same as [[Monte Carlo tree search|MCTS]] simulation: sample from $s_t$, backpropagate returns to estimate $Q(s_t)$. A [[Process Reward Models|PRM]] bridges these — it lets you evaluate intermediate reasoning states without rolling to completion, functioning as the value function at MCTS leaf nodes. This enables tree search during training: collect high-quality trajectories via MCTS + PRM, train policy on those. The bottleneck is PRM reliability — a misfit PRM is a [[Goodhart's Law|Goodhart]] problem at every step.

## General Applicability Beyond LLMs

GRPO is not LLM-specific. The pattern: fix a starting state $s_i$, "teleport" back to it, run $G$ rollouts, compute group-normalized advantages, update policy. LLMs make teleporting trivial (re-feed the prompt). In real environments you need a reset mechanism or world model — at which point this connects naturally to model-based RL.

> [!info] Why LLMs don't need model-based RL
> The "environment" (verifier / code interpreter) is essentially free to query, so the sample efficiency motivation for model-based RL largely disappears. The learned RM is also untrustworthy for imagined rollouts (Goodhart). The model-based analog for LLMs is test-time compute (beam search, MCTS at inference) rather than training-time world models.
