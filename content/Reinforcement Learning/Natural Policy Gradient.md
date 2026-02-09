---
aliases:
  - Natural Gradient
---

Source is the same as [[Policy Gradient]]

If you think about how we optimize gradient normally it actually doesn't make much sense: some parameters change probabilities a lot more than others, but we scale them the same (vanilla SGD)

The optimization can be seen as a constraint optimization problem:
$$\begin{aligned} \theta' \leftarrow \arg \max_{\theta'} (\theta' - \theta)^T \nabla_{\theta} J(\theta) \text{ s.t. } \underbrace{\|\theta' - \theta\|^2 \leq \epsilon}_{\text{controls how far we go}} \end{aligned}$$
In the 2D case, that's basically saying how far we should go on the x-axis so that the first-order gradient times $\delta x$ (which gives us y) descends the most. We do not want to go too far because the first-order approximation is only valid within a small range.

Really we should be constrained by "output does not change too much", not "all params in all dims does not change too much". 
$$\begin{aligned} &\theta' \leftarrow \arg \max_{\theta'} (\theta' - \theta)^T \nabla_{\theta} J(\theta) \text{ s.t. } D(\pi_{\theta'}, \pi_{\theta}) \leq \epsilon \\ &\quad \text{parameterization-independent divergence measure} \\ &\quad \text{usually KL-divergence: } D_{\text{KL}}(\pi_{\theta'} \| \pi_{\theta}) = E_{\pi_{\theta'}} [\log \pi_{\theta} - \log \pi_{\theta'}] \\ &D_{\text{KL}}(\pi_{\theta'} \| \pi_{\theta}) \approx (\theta' - \theta)^T \mathbf{F} (\theta' - \theta) \quad\quad \mathbf{F} = E_{\pi_{\theta}} [\nabla_{\theta} \log \pi_{\theta}(\mathbf{a}|\mathbf{s}) \nabla_{\theta} \log \pi_{\theta}(\mathbf{a}|\mathbf{s})^T] \\ &\quad \quad \quad \quad \text{Fisher-information matrix} \end{aligned}$$
So here comes [[KL Divergence]], if we Taylor expand that, we'll see the first two terms are all 0, and it can be approxiamated just by the second term, the Fisher-information matrix, and it's the Reimannian metric on the manifold of probability distributions. 
$$\begin{aligned} \theta' &\leftarrow \arg \max_{\theta'} (\theta' - \theta)^T \nabla_{\theta} J(\theta) \text{ s.t. } \|\theta' - \theta\|_{\mathbf{F}}^2 \le \epsilon \\ \theta &\leftarrow \theta + \alpha \mathbf{F}^{-1} \nabla_{\theta} J(\theta) \end{aligned}$$
We'll see later how [[TRPO]] is based on this idea.