---
original title: Proximal Policy Optimization Algorithms
Arxiv: https://arxiv.org/abs/1707.06347
pdf: "[[ppo.pdf]]"
year: 2017
date: 2026-04-19
---
I thought PPO is so daunting as it's used by every LLM and some robotics frontier research. It turns out it's actually much simpler than [[TRPO]], just like how [[GRPO]] is simpler than [[PPO]]. 

The paper itself is easy to read. Other than that there's also [OpenAI Spinning up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) which offers simpler formulas (why don't they include that in the paper? I derived them myself anyway since it was confusing) and explanations, and the [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/) on PyTorch blog. Note that blog contains invalid opinions, e.g. when explaining why there's the `min` in the PPO formula. I think the author confuses themselves. 

Anyway, let's get started!

---

The following note come from my conversation with Claude Sonnet 4.6

---

PPO asks the same question as [[TRPO]]: how do we take the biggest possible improvement step without accidentally collapsing performance? Where TRPO answers with complex second-order machinery, PPO is a family of **first-order methods** that enforce the trust region through the loss function shape itself — no Fisher matrix, no conjugate gradient, just plain SGD/Adam.

There are two variants: **PPO-Clip** (primary) and **PPO-Penalty** (adaptive KL). This note focuses on PPO-Clip.

## The Clipped Surrogate Objective

Let $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}$ be the probability ratio. $\theta_k$ is the **frozen behavioral policy** — the parameters that generated the current rollout batch $\mathcal{D}_k$; it stays pinned throughout all inner SGD epochs, with $k$ only incrementing when new trajectories are collected. The raw (unclipped) surrogate is just $r_t A_t$ — this is $L^{CPI}$, the policy gradient objective with importance sampling.

The PPO-Clip objective is:

$$
 L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ g(\epsilon, \hat{A}_t) \right) \right] 
$$

where the clipping function is:

$$
 g(\epsilon, A) = \begin{cases} (1 + \epsilon) A & A \geq 0 \\ (1 - \epsilon) A & A < 0 \end{cases} 
$$

This simplified form (from SpinningUp) is equivalent to the original paper's $\min(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$ but makes the intent clearer.

>[!question] Why `min()` on top of clipping?
> Clipping alone just flatlines the objective at the boundary — it stops rewarding further movement, but doesn't penalize overshooting. The `min()` makes it a **pessimistic lower bound**: it reintroduces the worse (unclipped) value whenever you've moved so far that it's worse than the clipped version.
> The two cases make this concrete:
>
> **When $A_t > 0$** (action was good, increase its probability):
> 
> - Past $(1+\epsilon)$, the clipped term flatlines
> - The unclipped term keeps growing — so `min` = clipped (flatline) ✓
> - No gradient incentive to keep pushing $r_t$ above $1+\epsilon$
> 
> **When $A_t < 0$** (action was bad, decrease its probability):
> 
> - Past $(1-\epsilon)$, the clipped term flatlines
> - The unclipped term keeps getting more negative — so `min` = unclipped (penalty) ✓
> - Overshoot is still penalized even though you're outside the clip band
> 
> Equivalently, you could write it with a conditional — only clip the side that would let the objective keep improving:
>
> $$
>  L_t = \begin{cases} \min(r_t,\ 1+\epsilon), A_t & A_t > 0 \\ \max(r_t,\ 1-\epsilon), A_t & A_t < 0 \end{cases} 
> $$
>
> The `min()` formulation is just a branchless way to express this.


## On the notation $L$: it's a pseudo-loss

$L^{CLIP}$, $L^{CPI}$, etc. are **surrogate objectives**, not the true RL objective. The true objective is the expected return $J(\theta)$; $L$ is a differentiable stand-in whose gradient is *engineered* to point in a useful direction. Optimizers minimize, so in code the policy "loss" is the **negated batch mean** of the surrogate:

$$
L_{\text{policy}} = -\frac{1}{N}\sum_t \min\!\big(r_t \hat{A}_t,\ g(\epsilon, \hat{A}_t)\big)
$$

The mathematical object has an expectation $\hat{\mathbb{E}}_t$; the minibatch mean is just its Monte Carlo estimate.

> [!important] Why no [[Score function|REINFORCE log-derivative trick]] here?
> The whole reason REINFORCE needs the trick is that its expectation is taken **over $\pi_\theta$ itself** — the sampling distribution depends on the very parameters we differentiate. You can't push $\nabla_\theta$ through $\mathbb{E}_{a\sim\pi_\theta}[\cdot]$ directly, so you rewrite $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$ to turn the gradient *back* into an expectation you can sample.
>
> PPO's surrogate is an expectation over the **frozen** $\pi_{\theta_k}$ (see [[Importance sampling corrections]]):
>
> $$ L^{CPI}(\theta) = \mathbb{E}_{(s,a)\sim\pi_{\theta_k}}\!\left[ r_t(\theta)\,\hat{A}_t \right] $$
>
> The distribution we sample from **does not depend on $\theta$**. So we can just sample directly and move the gradient inside — no trick needed:
>
> $$ \nabla_\theta L^{CPI} = \mathbb{E}_{\pi_{\theta_k}}\!\left[ \hat{A}_t\, \nabla_\theta r_t(\theta) \right] $$
>
> That's the conceptual leap from on-policy PG to importance-sampled surrogates: **fix the sampling distribution first, then correct the mismatch with the ratio $r_t$.**

> [!note] The score function reappears anyway
> Even though we never *invoke* the trick, differentiating the ratio reconstructs it. Since $r_t = \exp(\log\pi_\theta - \log\pi_{\theta_k})$ and $\pi_{\theta_k}$ is constant, $\nabla_\theta r_t = r_t\, \nabla_\theta \log\pi_\theta$. Evaluated at $\theta=\theta_k$ where $r_t=1$, the surrogate gradient collapses to $\mathbb{E}[\hat{A}_t \nabla_\theta \log\pi_\theta]$ — **exactly the policy gradient**. So PPO's gradient equals vanilla PG in a neighborhood of $\theta_k$; clipping only changes what happens once the inner SGD epochs drift $r_t$ far from 1.
>
> (Aside: the `exp(logp_new - logp_old)` in code is *not* logits→probabilities — that's softmax. It's recovering the probability *ratio* $\pi_\theta/\pi_{\theta_k}$ from log-probs in a numerically stable way.)

## Why This Works Better Than TRPO in Practice

- See [[TRPO#Limitations of TRPO]]

> [!question] But TRPO has monotonic improvement guarantees — how can PPO beat it?
> 
> The theoretical bound is extremely loose. What matters empirically is "does this update move in a good direction without destabilizing training?" Clipping is a robust heuristic for this. PPO also benefits from the *multiple SGD epochs* squeezing more signal from each batch of environment data, which TRPO simply can't do. The complexity TRPO pays for its guarantee buys very little in practice.
> I pressed Sonnet 4.6 really hard and it admit it's all empirical.

## PPO-Penalty (Adaptive KL)

Instead of clipping, penalize KL directly:

$$
 L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\left[ r_t \hat{A}_t - \beta \cdot D_{KL}[\pi_{\theta_k} | \pi_\theta] \right] 
$$

$\beta$ is adjusted each update:

- If $KL > 1.5,\delta$: increase $\beta$
- If $KL < \delta / 1.5$: decrease $\beta$
- Otherwise: leave it

> [!note] 
> This is basically bang-bang control with a deadband — a discrete switching rule on threshold crossings. The paper admits $\beta$-tuning is fiddly, and PPO-Clip empirically outperforms this variant. The more principled version of adaptive $\lambda$ is **dual gradient descent** (gradient ascent on the Lagrange multiplier), which has actual convergence theory and shows up in constrained RL (CMDPs) and RLHF KL tuning.

## Entropy Bonus

The full objective often adds an entropy term:

$$
 L = L^{CLIP} - c_1 L^{VF} + c_2 H[\pi_\theta] 
$$

where $H[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$.

This prevents **policy collapse** — gradient descent naturally wants to concentrate probability on the best-so-far action, which kills exploration. High entropy = spread distribution (exploratory). The entropy bonus penalizes overconfident policies, keeping exploration alive longer.

This is the same regularization intuition as **load balancing loss in [[Mixture of Experts]]**: the primary objective has no incentive to maintain diversity, so you add a term that explicitly fights the degenerate low-entropy solution. The difference is what "collapse" means — MoE collapses across experts for a token; policy gradient collapses across actions for a state.

```pseudo
\begin{algorithm}
\begin{algorithmic}
\REQUIRE initial policy parameters $\theta_{0}$, initial value function parameters $\phi_{0}$
\FOR{$k = 0, 1, 2, \dots$}
    \STATE Collect set of trajectories $\mathcal{D}_{k} = \{\tau_{i}\}$ by running policy $\pi_{k} = \pi(\theta_{k})$ in the environment.
    \STATE Compute rewards-to-go $\hat{R}_{t}$.
    \STATE Compute advantage estimates, $\hat{A}_{t}$ (using any method of advantage estimation) based on the current value function $V_{\phi_{k}}$.
    \STATE Update the policy by maximizing the PPO-Clip objective:
    \STATE $$\theta_{k+1} = \arg \max_{\theta} \frac{1}{|\mathcal{D}_{k}|T} \sum_{\tau \in \mathcal{D}_{k}} \sum_{t=0}^{T} \min \left( \frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{k}}(a_{t}|s_{t})} A^{\pi_{\theta_{k}}}(s_{t}, a_{t}), g(\epsilon, A^{\pi_{\theta_{k}}}(s_{t}, a_{t})) \right)$$
    \STATE \textit{typically via stochastic gradient ascent with Adam.}
    \STATE Fit value function by regression on mean-squared error:
    \STATE $$\phi_{k+1} = \arg \min_{\phi} \frac{1}{|\mathcal{D}_{k}|T} \sum_{\tau \in \mathcal{D}_{k}} \sum_{t=0}^{T} \left( V_{\phi}(s_{t}) - \hat{R}_{t} \right)^{2}$$
    \STATE \textit{typically via some gradient descent algorithm.}
\ENDFOR
\end{algorithmic}
\end{algorithm}
```
