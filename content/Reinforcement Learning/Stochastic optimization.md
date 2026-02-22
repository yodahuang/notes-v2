---
date: 2026-02-19
---
Pick some action and let's roll!
$$\begin{aligned} a_1,\ldots,a_T &\;=\; \arg\max_{a_1,\ldots,a_T}\; \underbrace{J\!\left(a_1,\ldots,a_T\right)}_{\text{don't care what this is}} \\ A &\;=\; \arg\max_{A}\; J(A) \end{aligned}$$
We can simply do random shooting (parallelize very well), or doing these:

[[Cross Entropy Method|CEM]]
[[Monte Carlo tree search]]

## Derivative based methods
$$\begin{aligned} &\min_{\mathbf{u}_1, \dots, \mathbf{u}_T} \sum_{t=1}^{T} c(\mathbf{x}_t, \mathbf{u}_t) \quad \text{s.t.} \quad \mathbf{x}_t = f(\mathbf{x}_{t-1}, \mathbf{u}_{t-1}) \\ &\min_{\mathbf{u}_1, \dots, \mathbf{u}_T} c(\mathbf{x}_1, \mathbf{u}_1) + c(f(\mathbf{x}_1, \mathbf{u}_1), \mathbf{u}_2) + \dots + c(f(f(\dots)\dots), \mathbf{u}_T) \end{aligned}$$
$2^{nd}$ order method tends to work better than first order method, as the chain can be long and it's easy to see vanishing / exploding gradients.

Shooting vs collocation method: we can either optimize over actions only, or both action and state with constraints (conditions better)
$$\min_{\mathbf{u}_1, \dots, \mathbf{u}_T, \mathbf{x}_1, \dots, \mathbf{x}_T} \sum_{t=1}^T c(\mathbf{x}_t, \mathbf{u}_t) \quad \text{s.t.} \quad \mathbf{x}_t = f(\mathbf{x}_{t-1}, \mathbf{u}_{t-1})$$

In the linear case, we have a nice dynamic programming thing: [[Linear-quadratic regulator|LQR]]]