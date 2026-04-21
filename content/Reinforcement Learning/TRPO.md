---
date: 2026-02-18
pdf: "[[285-advanced-policy-gradient.pdf]]"
original title: Trust Region Policy Optimization
year: 2015
---

The following stuff is from CS285, lecture 9.

One could also think about [[Policy Gradient]] as a soft form of [[Policy & value iteration|Policy Iteration]]. Instead of directly changing policy according to the belief of advantage, we just adjust the policy a bit.

This idea would lead to [[TRPO]], [[PPO]] and others.

As policy iteration update $\theta$ to be the new $\theta'$, policy iteration is basically optimizing this in the policy improvement part:

$$
\mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[ \sum_t \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
$$

Why? This is "pick the best action for each state, based on previous belief of the value functions".

Recall the real RL objective $J(\theta)$, we can show that (process omitted): 

$$
\begin{aligned}
J(\theta) &= \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_t \gamma^t r(s_t, a_t) \right] \\
J(\theta') - J(\theta)&=\mathbb{E}_{\tau \sim p_{\theta'}(\tau)}  
\left[  
\sum_{t} \gamma^t A^{\pi_\theta}(s_t, a_t)  
\right] \\
&= \sum_{t}  
\mathbb{E}_{s_t \sim p_{\theta'}(s_t)}  
\left[  
\mathbb{E}_{a_t \sim \pi_{\theta}(a_t \mid s_t)}  
\left[  
\frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_{\theta}(a_t \mid s_t)}  
\gamma^t  
A^{\pi_\theta}(s_t, a_t)  
\right]  
\right]  
\end{aligned}
$$

This means optimizing the RL objective *IS* the policy iteration objective.
And if we swap that $p_{\theta'}$ to $p_{\theta}$ for $s_t$, we just have our off policy policy gradient back exactly. If we evaluate at $\theta = \theta'$, we got policy gradient back.

>[!question] Wait, isn't policy gradient also directly optimizing RL objective?
> You might be wondering why both are optimized directly on the RL objective, but they are different. The reason is that the policy gradient objective is really a **first-order approximation**. It's an infinitesimal policy iteration. That's why if you evaluate it at $\theta = \theta'$ we get policy gradient objective, a surrogate objective.

Now, why would we want to swap that $p_{\theta'}$ to $p_{\theta}$ for $s_t$? If you think about how we really train our model, what we have is current policy, not the "next policy" as we do optimization, and that's how policy gradient work. And that makes things pretty simple.

*Claim*: $p_\theta(s_t)$ is close to $p_{\theta'}(s_t)$ when $\pi_\theta$ is close to $\pi_{\theta'}$.

We can prove it the same way we get the error bound of [[Imitation Learning]]. The core idea is that suppose the policies are different by [total variation distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) $\epsilon$, and we can show $\sum_{t}E_{s_{t}\sim \pi_{\theta}(a_t|s_t)}$ is close.

So we can now say: yes this can be swapped as long as $\pi_{\theta}$ is close.

$$
\begin{aligned}
\theta' \leftarrow \arg\max_{\theta'} \quad &
\sum_{t}
\mathbb{E}_{s_t \sim p_{\theta}(s_t)}
\left[
\mathbb{E}_{a_t \sim \pi_{\theta}(a_t \mid s_t)}
\left[
\frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_{\theta}(a_t \mid s_t)}
\, \gamma^t A^{\pi_\theta}(s_t, a_t)
\right]
\right]
\\
\text{such that} \quad &
D_{\mathrm{KL}}\!\left(\pi_{\theta'}(a_t \mid s_t)\,\|\,\pi_{\theta}(a_t \mid s_t)\right)
\le \epsilon
\end{aligned}
$$

This is now very similar to [[Natural Policy Gradient|Natural Gradient]], but hey we are not there yet.

> [!tip] Quick primer of constrained optimization
>
> Suppose you want to solve:
>
> $$
> \max_{\theta'} f(\theta') \quad \text{s.t.} \quad g(\theta') \le \varepsilon
> $$
>
> This is a **constrained optimization problem**.
> Instead of optimizing inside a hard constraint region, we introduce a penalty variable (the Lagrange multiplier) $\lambda \ge 0$  and form:
>
> $$
> \mathcal{L}(\theta', \lambda) = 
> f(\theta') -
> \lambda (g(\theta') - \varepsilon)
> $$
>
> Interpretation:
> - If the constraint is satisfied, fine.
> - If $g(\theta') > \varepsilon$, the penalty term becomes active.
> - $\lambda$ adjusts how harshly we penalize constraint violation.
>
> So instead of solving a constrained problem directly, we solve a **saddle point problem**:
>
> $$
> \max_{\theta'} \min_{\lambda \ge 0} \mathcal{L}(\theta', \lambda)
> $$
>
> This is called the **primal–dual formulation**. It's only exact under certain convexity conditions. It always gives an upper bound on the primal optimum.

Now since we are dealing with NN, we can't solve it nicely. We can use dual gradient descent for this.

Or we can just do second order Taylor expansion and approximate $KL$ by $F$, [[Fisher information]], which leads to [[Natural Policy Gradient|Natural Gradient]]. This is doable since now we convert $KL$ to a quadratic function: $\frac{1}{2}\Delta\theta^{T}H\Delta\theta$, which makes it easy to do $\nabla_{\theta'}$, solvable in closed form.

If we solve that we got

$$
\begin{aligned}
\theta' &= \theta + \alpha F^{-1}\nabla_{\theta} J(\theta) \\
\alpha &= \sqrt{\frac{2\epsilon}{\nabla_{\theta} J(\theta)^{T} F \nabla_{\theta} J(\theta)}}
\end{aligned}
$$

There's more in TRPO on how to do efficient Fisher-vector products.

---

The following is from a conversation with Claude Sonnet 4.6, when reading [[PPO]] paper.

---

## Efficient Fisher-Vector Products via Conjugate Gradient

$F$ is $|\theta| \times |\theta|$ — impossible to store or invert for any real network. Instead, TRPO reformulates $F^{-1}g$ as solving the linear system $Fx = g$ using **conjugate gradient (CG)**, which only needs matrix-vector products $Fv$, never $F$ itself.

Computing $Fv$ for arbitrary $v$ is cheap. Since $F = \mathbb{E}[\nabla \log \pi \cdot \nabla \log \pi^T]$:

$$
Fv = \mathbb{E}[(\nabla \log \pi^T v)\nabla \log \pi]
$$

This requires two backward passes (or one forward-over-backward autodiff pass) — no $|\theta|^2$ storage. CG runs ~10 iterations, each needing one $Fv$ product, to approximate $F^{-1}g$. Then $\alpha$ is computed from the closed-form formula above.

> [!note] Cost
> 
> ~10x more expensive than a plain SGD step, plus requires custom autodiff infrastructure. This is exactly what [[PPO]] wanted to escape.

## Limitations of TRPO

**Incompatible with parameter sharing.** The KL constraint is defined purely over policy outputs. When policy and value function share parameters, a gradient step satisfying the KL constraint may cause a large unconstrained update to the value head — or the value loss gradient may yank shared parameters in a way that violates the policy's KL budget. There's no clean way to enforce the constraint on the policy while jointly optimizing a value loss through shared weights.

**Incompatible with dropout.** CG requires multiple forward passes to compute Fisher-vector products. Dropout samples a different mask each pass, so each CG iteration is computing $Fv$ for a _different network_. CG's convergence guarantee assumes repeated multiplication by the _same_ matrix $F$ — dropout breaks this structurally, not just noisily. Freezing the mask is a workaround but adds yet more implementation complexity.

**The approximation is imperfect anyway.** The Fisher approximation (quadratic KL) and finite CG iterations mean TRPO doesn't actually solve the constrained problem exactly. The complexity cost is real; the theoretical guarantee is approximate.

> [!question] Where does the penalty form $\beta \cdot \text{KL}$ come from?
> 
> The theory (an error bound proof, analogous to [[Imitation Learning]]) shows there _exists_ some $\beta$ such that optimizing surrogate $- \beta \cdot \text{KL}$ guarantees monotonic improvement. The $\beta$ absorbs horizon, discount, and bound constants — it's problem-dependent and changes over training. This is why TRPO uses the hard constraint form instead: $\epsilon$ is far more stable to tune than $\beta$. See [[PPO]] for how this tension is resolved differently.