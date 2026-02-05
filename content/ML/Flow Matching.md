---
date: 2026-02-03
---

## Source

This is all from course material of Introduction to Flow Matching and Diffusion Models. MIT Computer Science Class 6.S184: Generative AI with Stochastic Differential Equations.
- Class website: https://diffusion.csail.mit.edu/2026/index.html
- Class notes: [[flow_matching_diffusion_notes.pdf]]. I did the 2025 version and the note is from 2026. I think the only new thing is the CTMC models.
- Appreciate the course as all the slides and recordings are released with lab and solutions

Take the note as the ground truth, here I'll just list out some brief idea for quick overview.

The goal is to go from $x \sim p_{init}$ to $z \sim p_{data}$. There are several ways to do that. If we try to predict directly, that will be GAN. The problem is that the training is unstable because the reward is sparse.

So imagine for example, if we want to convert a pile of dirt into a mountain, someone just telling you that you missed or hit it is not enough. Flow matching is basically saying, “Why don’t we provide guidance in the process?” so the whole thing is easier to fit.

We could say that the flow-match algorithm basically converts a generation problem into a supervised learning problem; you are essentially supervised on the given velocity path that we know would work.

There is another critical reason why the flow-matching algorithm is so great: it only needs several trajectories that could work. It doesn't require a single, definitive trajectory; it can simply generate a multi-hypothesis output because of a very neat mathematical property:

A **flow model** is then described by the ODE

$$\begin{aligned} X_0 &\sim p_{\text{init}} && \blacktriangleright \text{ random initialization} \\ \frac{\mathrm{d}}{\mathrm{d}t} X_t &= u_t^\theta(X_t) && \blacktriangleright \text{ ODE} \end{aligned}$$

Our goal is to make the endpoint $X_1$ of the trajectory have distribution $p_{\text{data}}$, i.e.

$$X_1 \sim p_{\text{data}} \quad \Leftrightarrow \quad \psi_1^\theta(X_0) \sim p_{\text{data}}$$

**Conditional flow matching loss** and **marginal flow matching loss** differ only by a constant (w.r.t. θ),  so they have the _same minimizer_.

So, what one would do is generate training data. We generate it in a supervised learning way, which is basically sampling here, because you just need to sample:
1. Your z
2. Your time step
3. Your noise (Gaussian noise or something)
4. Calculate $x$ based on these. E.g. $x = tz + (1 -t)\epsilon$.

We just need to make sure the probability path we choose converge to $z$ in the end and is $x_0$ in the beginning.

We can derive the expected vector field $u_t$ easily, provided that we choose a path. For example:
let $p_t(\cdot|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$ for noise schedulers $\alpha_t, \beta_t$ . Let $\dot{\alpha}_t = \partial_t \alpha_t$ and $\dot{\beta}_t = \partial_t \beta_t$ denote respective time derivatives of $\alpha_t$ and $\beta_t$. he conditional Gaussian vector field is given by

$$u_t^{\text{target}}(x|z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x$$

This can be derived on demand:
The conditional flow model is $\psi_t^{\text{target}}(x|z) = \alpha_t z + \beta_t \epsilon$. You just do derivative on $t$ to the components, and replace $\epsilon$ with $x$ and $z$ since the value of that is determined since we already sampled $x$. Or you can just say it's $\dot{\alpha_{t}}z + \dot{\beta_{t}}\epsilon$ . 

```pseudo
\begin{algorithm}
\caption{Flow Matching Training Procedure (for Gaussian CondOT path $p_t(x|z) = \mathcal{N}(tz, (1 - t)^2)$)}
\begin{algorithmic}
\REQUIRE A dataset of samples $z \sim p_{\text{data}}$, neural network $u_t^\theta$
\FOR{each mini-batch of data}
    \STATE Sample a data example $z$ from the dataset.
    \STATE Sample a random time $t \sim \text{Unif}_{[0,1]}$.
    \STATE Sample noise $\epsilon \sim \mathcal{N}(0, I_d)$
    \STATE Set
    \[
    x = tz + (1 - t)\epsilon \quad \quad \text{(General case: } x \sim p_t(\cdot | z)\text{)}
    \]
    \STATE Compute loss
    \[
    \mathcal{L}(\theta) = \|u_t^\theta(x) - (z - \epsilon)\|^2 \quad \quad \text{(General case: } = \|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2\text{)}
    \]
    \STATE Update $\theta \leftarrow \text{grad\_update}(\mathcal{L}(\theta))$.
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

```pseudo
\begin{algorithm}
\caption{Sampling from a Flow Model with Euler method}
\begin{algorithmic}
\REQUIRE Neural network vector field $u_t^\theta$, number of steps $n$
\STATE Set $t = 0$
\STATE Set step size $h = \frac{1}{n}$
\STATE Draw a sample $X_0 \sim p_{\text{init}}$ \COMMENT{Random initialization!}
\FOR{$i = 1, \dots, n-1$}
    \STATE $X_{t+h} = X_t + h u_t^\theta(X_t)$
    \STATE Update $t \leftarrow t + h$
\ENDFOR
\RETURN $X_1$ \COMMENT{Return final point}
\end{algorithmic}
\end{algorithm}
```
