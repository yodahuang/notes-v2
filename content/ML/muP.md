---
date: 2026-04-05
pdf: "[[spectral_condition_feature_learning.pdf]]"
original title: A Spectral Condition for Feature Learning
---
Disclaimer: I did not read the OG paper. This is from [[cs336_lecture_11.pdf]], as well as some chat with Gemini 3 Pro.

---

This is the "by-product" of Greg Yang's Tensor Program series. It's actually first discussed in the paper "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" and then a more accessible version "A Spectral Condition for Feature Learning"

See [The series](https://thegregyang.com/#tppapers)

> You can’t train GPT-3 on a single GPU, much less tune its hyperparameters (HPs).
> 
> But what if I tell you…
> 
> …you *can* tune its HPs on a single GPU thanks to the theory developed in TP4?
> 
> Essentially, narrow and wide neural networks share the same set of optimal hyperparameters if they are in the maximal update parametrization (muP) derived in TP4 (but not if they are in pytorch default parametrization).

muP is based off the following assertion. As a function of the width of the network $n_l$...
A1: The activations at initialization should remain $\Theta(1)$
A2: After one gradient step, the change in activation should be $\Theta(1)$

> [!info] What's $\Theta$
> A tight bound, "exactly this scale", unlike $O$. So something that's $\Theta(1)$ cannot be $\frac{1}{n}$, for example, as it vanishes.

## Spectral Norm

From conversation with Claude Sonnet 4.6.
### Back to SVD
Any matrix $W \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$
W = U \Sigma V^\top
$$

where:

- $U \in \mathbb{R}^{m \times m}$ — orthogonal matrix (output directions)
- $V \in \mathbb{R}^{n \times n}$ — orthogonal matrix (input directions)
- $\Sigma$ — diagonal matrix with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$

Those diagonal entries $\sigma_i$ are the **singular values**.

Every matrix is just "rotate → stretch → rotate":

$$
x \xrightarrow{V^\top} \text{rotate input} \xrightarrow{\Sigma} \text{stretch each axis} \xrightarrow{U} \text{rotate output}
$$

The singular values are the **stretch factors** along each axis. So:

- $\sigma_{\max}$ = the most any direction gets stretched → that's $|W|_2$
- $\sigma_{\min}$ = the most any direction gets squished

And the spectral norm is just the largest singlular value, it directly measures the maximum amplification a layer applies to any input direction.

The paper shows that the training can be stable if that spectral norm

$$
||w|| = \Theta(\sqrt{\frac{\text{fan}_\text{out}}{\text{fan}_{\text{in}}}})
$$

and then some more derivations.
## In practice

$$
\begin{aligned} \textbf{Initialization: } & \text{Set to } \Theta \left( \frac{1}{\sqrt{n_{l-1}}} \min \left( 1, \sqrt{\frac{n_l}{n_{l-1}}} \right) \right) \\ \textbf{Learning rates: } & \text{Set to } \frac{n_l}{n_{l-1}} \quad \text{(for Adam } \frac{1}{n_{l-1}}\text{)} \end{aligned}
$$

[[Cerebras GPT]] uses this and offers nice tables.

I find this implementation quite interesting: [ezmup](https://github.com/cloneofsimo/ezmup). Different from the official implementation:

> First, in your project, change the config so that some weird large enough prime number represents the _varying width_. By _weird_, I mean such number should not be used in your other hyperparameters of model shapes. 47 is such good number.


## What's it not robust to
- Exotic optimizers
- (strong) weight decay

