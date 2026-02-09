---
pdf: "[[cs285_policy_gradient.pdf]]"
aliases:
  - REINFORCE
---
Source: The 2023 version of CS285, lecture 5.

## REINFORCE

Policy gradient is direct optimization: for [[Function approximation]] we are just going to find the $\theta$ such that:

$$
\theta^{\star} = \arg \max_{\theta} \underbrace{E_{\tau \sim p_{\theta}(\tau)} \left[ \sum_{t} r(\mathbf{s}_t, \mathbf{a}_t) \right]}_{J(\theta)}
$$

$$
J(\theta) = E_{\tau \sim p_{\theta}(\tau)} [\overbrace{r(\tau)}^{\sum_{t=1}^{T} r(\mathbf{s}_t, \mathbf{a}_t)}] = \int p_{\theta}(\tau) r(\tau) d\tau
$$

So $\tau$ here is basically the whole trajectory. We are just trying to compute the derivative of $J(\theta)$ w.r.t. $\theta$. Now, there is this expectation here, and it's hard for us to get the gradient on that. 

$$
\nabla_{\theta} J(\theta) = \int \nabla_{\theta} p_{\theta}(\tau) r(\tau) d\tau = \int p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) r(\tau) d\tau = E_{\tau \sim p_{\theta}(\tau)} [\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)]
$$

This is super nice, since well, we basically want to have a $p_{\theta}(\tau)$ inside that $\int$ . This is because well there's a convenient property:

$$
p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) = p_{\theta}(\tau) \frac{\nabla_{\theta} p_{\theta}(\tau)}{p_{\theta}(\tau)} = \nabla_{\theta} p_{\theta}(\tau)
$$

Let's  then look at that $\log p_{\theta}(\tau)$. What's that exactly?

$$
\nabla_{\theta} \left[ \cancel{\log p(\mathbf{s}_1)} + \sum_{t=1}^{T} \log \pi_{\theta}(\mathbf{a}_t|\mathbf{s}_t) + \cancel{\log p(\mathbf{s}_{t+1}|\mathbf{s}_t, \mathbf{a}_t)} \right]
$$

The parts are canceled since they are not related to $\theta$. Finally, we got

$$
\nabla_{\theta} J(\theta) = E_{\tau \sim p_{\theta}(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_t|\mathbf{s}_t) \right) \left( \sum_{t=1}^{T} r(\mathbf{s}_t, \mathbf{a}_t) \right) \right]
$$

*REINFORCE* algorithm:

1. sample $\{\tau^i\}$ from $\pi_{\theta}(\mathbf{a}_t|\mathbf{s}_t)$ (run the policy)
2. $\nabla_{\theta} J(\theta) \approx \sum_{i} \left( \sum_{t} \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}_t^i|\mathbf{s}_t^i) \right) \left( \sum_{t} r(\mathbf{s}_t^i, \mathbf{a}_t^i) \right)$
3. $\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)$

We are seeing $\nabla_{\theta}\log p_{\theta}(x)$ a lot recently! See [[Score function]] note for more.

### More interpretation

Now look at Maximum likelihood, say we are doing [[Imitation Learning]]:

$$
	\nabla_{\theta}J_{ML}(\theta)\approx\frac{1}{N}\sum^N_{i=1}(\sum^T_{t=1}\nabla_{\theta}\log\pi_{\theta}(a_{i,t}|s_{i,t}))
$$

Well that's very much just our REINFORCE formula, but just without the $r$ bit. You can say policy gradient is weighting the experience with the reward received, trial and error.

Note that the Markov property is not exploited in the derivation.

## Problems

Note that we are rating the trajectory via reward. If we have three samples, `[-2, 1, 2]`, and that should be equivalent to `[1, 4, 5]` (see baseline later), but variance wise that's different. With finite samples, these do mean different things to the estimation. 

### Causality

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t} | \mathbf{s}_{i,t}) \right) \left( \sum_{t=1}^T r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}) \right)
$$

_Causality_: policy at time $t'$ cannot affect reward at time $t$ when $t < t'$

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t} | \mathbf{s}_{i,t}) \underbrace{\left( \sum_{t'=t}^T r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right)}_{\substack{\text{"reward to go"} \\ \hat{Q}_{i,t}}}
$$

We are writing out the expectation because then we can clearly see that one trajectory is really from 1 to time $T$, and use the distributive law. That $Q$ is a hint to the value function

### Baseline

Note $E[\nabla_\theta\log p_{\theta}(\tau)b] = 0$, so we can just reduce a baseline to the rewards

$$
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} \log p_{\theta}(\tau) [r(\tau) - b]
$$

We can get it by average or other ways, for example, just find out what's the best $b$, solving $\frac{dVar}{db} = 0$. We can get that $b = \frac{E[g(\tau)^2r(\tau)]}{E[g(\tau)^2]}$, expected reward weighted by gradient magnitudes.

## Off Policy

Refer to [[On and off policy Learning|Off-Policy]] for some background, and we know a way to make on-policy method work for off policy is [[Importance sampling corrections]]

$$
J(\theta') = E_{\tau \sim p_{\theta}(\tau)} \left[ \frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)} r(\tau) \right]
$$

$$
\begin{aligned}
\nabla_{\theta'} J(\theta') &= E_{\tau \sim p_{\theta}(\tau)} \left[ \frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)} \nabla_{\theta'} \log \pi_{\theta'}(\tau) r(\tau) \right] \quad \text{when } \theta \neq \theta' \\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ \left( \prod_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t)}{\pi_{\theta}(\mathbf{a}_t|\mathbf{s}_t)} \right) \left( \sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t) \right) \left( \sum_{t=1}^{T} r(\mathbf{s}_t, \mathbf{a}_t) \right) \right] \\
&= E_{\tau \sim p_{\theta}(\tau)} \left[ \sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t) \left( \underline{\prod_{t'=1}^{t} \frac{\pi_{\theta'}(\mathbf{a}_{t'}|\mathbf{s}_{t'})}{\pi_{\theta}(\mathbf{a}_{t'}|\mathbf{s}_{t'})}} \right) \left( \sum_{t'=t}^{T} r(\mathbf{s}_{t'}, \mathbf{a}_{t'}) \left( \cancel{\prod_{t''=t}^{t'} \frac{\pi_{\theta'}(\mathbf{a}_{t''}|\mathbf{s}_{t''})}{\pi_{\theta}(\mathbf{a}_{t''}|\mathbf{s}_{t''})}} \right) \right) \right]
\end{aligned}
$$

The last one is taking causality into account. The can be crossed out for simplicity, which makes it more like [[Policy & value iteration]]

In practice that big $\prod$ term is not great, since it tends to be smaller than 1 for each of the term (we are not selecting the action that's according to the policy), and it got exponentially small in as $T$ goes.

Now we can write the objective a bit differently to solve this: let's expand that expectation to sampling:

on-policy policy gradient, sampling form:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})\hat{Q}_{i,t}
$$

This can be understood as sampling at each step given the state each step. In other words, we took the on-policy gradient expressed as an expectation over the marginal state-action distribution at each timestep, and then we can apply the importance sampling on each marginal $(s_{t}, a{t})$ separately, correcting from $p_{\theta}(s_{t,}a_t)$ to  $p_{\theta'}(s_{t,}a_t)$ .

off-policy policy gradient:

$$
\begin{aligned} \nabla_{\theta'} J(\theta') &\approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \frac{\pi_{\theta'}(\mathbf{s}_{i,t}, \mathbf{a}_{i,t})}{\pi_{\theta}(\mathbf{s}_{i,t}, \mathbf{a}_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})\hat{Q}_{i,t} \\ &= \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \frac{\pi_{\theta'}(\mathbf{s}_{i,t})}{\pi_{\theta}(\mathbf{s}_{i,t})} \frac{\pi_{\theta'}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})}{\pi_{\theta}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})\hat{Q}_{i,t} \end{aligned}
$$

## Implementing

Well when doing this in a framework we'll work directly with $J$, not $\nabla J$, but our modifications are all on $\nabla J$, so now we'll need to have a "peudo-loss":

$$
\tilde{J}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log \pi_{\theta}(\mathbf{a}_{i,t} | \mathbf{s}_{i,t}) \hat{Q}_{i,t} 
$$

- Use much larger batches
- Tweaking learning rate is very hard

See also
[[Natural Policy Gradient]]