We can approximate the true value of a state by using function approximation. And we can find the best one using classic method. 

### Approximate state value function

Like if we use the very basic SGD, we just optimize this,

$$
J(\mathbf{w}) = \mathbb{E}_{S \sim d} \left[ \left( v_{\pi}(S) - v_{\mathbf{w}}(S) \right)^2 \right]
$$

and we can sample the gradient by 

$$
\Delta \mathbf{w} = \alpha \left( G_t - v_{\mathbf{w}}(S_t) \right) \nabla_{\mathbf{w}} v_{\mathbf{w}}(S_t)
$$

For encoding the state we can use something like [[Coarse coding]] (for simple linear approximation).

The $G_t$ here can be replaced by TD target $R_{t+1} + \gamma v_{w}(S_{t+1})$.

### Approximate state-action value function

There can be two ways to this: action-in, or action-out. For example (the linear case),
- Action in: $q_{\mathbf{w}}(s, a) = \mathbf{w}^\top \mathbf{x}(s, a)$  
- Action out: $\mathbf{q}_{\mathbf{w}}(s) = \mathbf{W} \mathbf{x}(s)$ such that $q_{\mathbf{w}}(s, a) = \mathbf{q}_{\mathbf{w}}(s)[a] = x(s)^Tw_a$
I think this can be viewed just from compute complexity point of view.
- Action in: $x(s, a)$ has dimension of $(s + a, 1)$. Thus $w$ has $(1, s+a)$ and we do $(s+a)$ element-wise multiplication. The same $w$ is used across all $(s, a)$ pair. But each state / action pair has different features.
- Action out: $w$ has dimension of $(a, s)$. We do $s * a$ element-wise multiplication. Multi-head parameterization. We use different $w$ for different action. But the actions from the same state shares the same feature. 

Action in is better for continuous actions (look at action out and you can see why).
Action out is more efficient for (small) discrete action spaces (I guess since it's more expressive).

### Convergence and Divergence

MC: this is easy. This can be seen as a simple regression: we have ground truth ($G_t$), and there is close form solution for linear regression case. Not so simple for TD.

$$
\begin{aligned}
\mathbf{w}_{\mathrm{MC}} & = \arg\min_{\mathbf{w}} \mathbb{E}_{\pi}\!\left[ \big(G_t - v_{\mathbf{w}}(S_t)\big)^2 \right]
&& = \mathbb{E}_{\pi}\!\left[ \mathbf{x}_t \mathbf{x}_t^{\top} \right]^{-1} \mathbb{E}_{\pi}\!\left[ G_t \mathbf{x}_t \right] \\[6pt]
\mathbf{w}_{\mathrm{TD}} & & &= \mathbb{E}\!\left[ \mathbf{x}_t \big(\mathbf{x}_t - \gamma \mathbf{x}_{t+1}\big)^{\top} \right]^{-1}
\mathbb{E}\!\left[ R_{t+1} \mathbf{x}_t \right]
\end{aligned}
$$

Let $\overline{\mathrm{VE}}(\mathbf{w})$ denote the **value error**:

$$
\overline{\mathrm{VE}}(\mathbf{w}) 
= \| v_{\pi} - v_{\mathbf{w}} \|_{d_{\pi}} 
= \sum_{s \in \mathcal{S}} d_{\pi}(s) \big( v_{\pi}(s) - v_{\mathbf{w}}(s) \big)^2
$$

The Monte Carlo solution minimises the value error
**Theorem**

$$
\overline{\mathrm{VE}}(\mathbf{w}_{\mathrm{TD}}) 
\le 
\frac{1}{1 - \gamma} \, \overline{\mathrm{VE}}(\mathbf{w}_{\mathrm{MC}})
= 
\frac{1}{1 - \gamma} \min_{\mathbf{w}} \overline{\mathrm{VE}}(\mathbf{w})
$$

So TD error is bounded.

Still, TD update is not a true gradient update: it includes itself in the other side.

#### Deadly triad

Also a name from Sutton and Barto's textbook.
Algorithms that combine
- bootstrapping
- off-policy learning
- function approximation
... may diverge.