---
date: 2026-02-03
---

## Source

This is all from course material of Introduction to Flow Matching and Diffusion Models. MIT Computer Science Class 6.S184: Generative AI with Stochastic Differential Equations.
- Class website: https://diffusion.csail.mit.edu/2026/index.html
- Class notes: [[flow_matching_diffusion_notes.pdf]]. I did the 2025 version and the note is from 2026. I think the only new thing is the CTMC models.
- Appreciate the course as all the slides and recordings are released with lab and solutions

Take the note as the ground truth, here I'll just list out some brief idea for quick overview.

---

Some parts of the note below comes from my conversation with Claude Sonnet 4.6 when reading [[Stable Diffusion 3]] paper. Most of the notes are hand written.

---

The goal is to go from $x \sim p_{init}$ to $z \sim p_{data}$. There are several ways to do that. If we try to predict directly, that will be GAN. The problem is that the training is unstable because the reward is sparse.

So imagine for example, if we want to convert a pile of dirt into a mountain, someone just telling you that you missed or hit it is not enough. Flow matching is basically saying, “Why don’t we provide guidance in the process?” so the whole thing is easier to fit.

We could say that the flow-match algorithm basically converts a generation problem into a supervised learning problem; you are essentially supervised on the given velocity path that we know would work.

There is another critical reason why the flow-matching algorithm is so great: it only needs several trajectories that could work. It doesn't require a single, definitive trajectory; it can simply generate a multi-hypothesis output because of a very neat mathematical property. We'll see that later.

---

A **flow model** is then described by the ODE

$$
\begin{aligned} X_0 &\sim p_{\text{init}} && \blacktriangleright \text{ random initialization} \\ \frac{\mathrm{d}}{\mathrm{d}t} X_t &= u_t^\theta(X_t) && \blacktriangleright \text{ ODE} \end{aligned}
$$

Our goal is to make the endpoint $X_1$ of the trajectory have distribution $p_{\text{data}}$, i.e.

$$
X_1 \sim p_{\text{data}} \quad \Leftrightarrow \quad \psi_1^\theta(X_0) \sim p_{\text{data}}
$$

The problem: to train $u^\theta_t$, we'd need to supervise it on the **marginal velocity field**:

$$
u_t(x) = \mathbb{E}[u^*_t(x_t | z) \mid x_t = x]
$$

This is intractable — it requires averaging over all $z \sim p_\text{data}$ consistent with $x_t$, which we can't compute.

---
## The Key Theorem: Conditional = Marginal

The neat property that makes flow matching work: the **conditional** and **marginal** losses have the same minimizer (differ only by a constant w.r.t. $\theta$):

$$
\mathcal{L}_\text{FM} = \mathbb{E}\|u^\theta_t(x_t) - u_t(x_t)\|^2 \quad \longleftrightarrow \quad \mathcal{L}_\text{CFM} = \mathbb{E}\|u^\theta_t(x_t) - u^*_t(x_t|z)\|^2
$$

So instead of supervising on the intractable marginal, we supervise on the **conditional velocity** $u^*_t(x_t|z)$ — which we *can* compute because we sampled $z$ from the dataset. See [[flow_matching_diffusion_notes.pdf#page=20&selection=370,0,370,10|notes p.20]] for the proof.

This also means: we don't need a single correct trajectory. Any set of conditional paths that reach $z$ at $t=1$ will do — the model learns to average over them.

---

## Constructing the Conditional Path

To compute $u^*_t(x_t|z)$, we need to define a path from noise to data. We build an interpolant:

$$
x_t = \alpha_t z + \beta_t \epsilon, \qquad \epsilon \sim p_\text{init}
$$

Boundary conditions: $\alpha_0 = 0,\ \alpha_1 = 1,\ \beta_0 = 1,\ \beta_1 = 0$ — pure noise at $t=0$, pure data at $t=1$.

When $p_\text{init} = \mathcal{N}(0,I)$ and the interpolant is affine like this, the conditional distribution is Gaussian:

$$
p_t(x|z) = \mathcal{N}(\alpha_t z,\ \beta_t^2 I)
$$

This is the **Gaussian CondOT path**. "Gaussian" means the conditional path is Gaussian. "CondOT" refers to the coupling — more below.

> [!note] Why Gaussian? The closed-form chain
> Because $\psi_t(x|z) = \alpha_t z + \beta_t \epsilon$ is affine, you can differentiate it directly w.r.t. $t$ to get $u^*$ in closed form. That's the whole point: Gaussian path → affine $\psi_t$ → $u^* = \dot\psi_t$ by calculus. Non-Gaussian paths exist (stochastic interpolants, discrete CTMC) but lose this.
>
>  "CondOT" refers to the **coupling**: how to pair $\epsilon$'s with $z$'s. Simplest: sample independently. This causes trajectories to cross, making $u_t(x)$ a blurry average. Alternative: **minibatch OT coupling** — within each training batch, instead of using $(z_i, \epsilon_i)$ pairs in random sampling order, solve a small assignment problem to find the permutation $\sigma$ that minimises $\sum_i \|z_i - \epsilon_{\sigma(i)}\|^2$. This pairs each data point with the geometrically closest noise sample, giving shorter and less curved trajectories. Fewer crossings → less ambiguity in $u_t(x)$ → easier to learn. It's the same minibatch gradient descent loop — just a smarter pairing step inside each batch. In practice the benefit is modest; SD3 and Flux use independent sampling and it works fine.

Different methods just pick different $(\alpha_t, \beta_t)$:

| Method | $\alpha_t$ | $\beta_t$ | Notes |
|---|---|---|---|
| **Rectified Flow** (SD3, Flux) | $t$ | $1-t$ | Linear — straight-line paths |
| DDPM (VP) | $\sqrt{\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}$ | Time runs *backward* (0=data, T=noise) |
| EDM | $1$ | $\sigma_t$ | Signal never scaled; $\sigma\to\infty$ = noise |

> [!note] DDPM convention
> DDPM flips the time direction: $t=0$ is clean data, $t=T$ is noise. Boundary conditions are reversed. Underneath, all methods trace the same SNR range — different parameterizations of the same transport.

---

## Deriving the Conditional Velocity $u^*$

Differentiate the interpolant w.r.t. $t$:

$$
\dot x_t = \dot\alpha_t z + \dot\beta_t \epsilon
$$

This has two unknowns ($z$ and $\epsilon$), linked by $x_t = \alpha_t z + \beta_t \epsilon$. Eliminating $\epsilon = (x_t - \alpha_t z)/\beta_t$:

$$
u^*_t(x_t|z) = \left(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\right)z + \frac{\dot\beta_t}{\beta_t}x_t
$$

### Rectified Flow specifically

$\alpha_t = t$, $\beta_t = 1-t$, $\dot\alpha_t = 1$, $\dot\beta_t = -1$. Substituting $x_t = tz + (1-t)\epsilon$:

$$
u^*_t = \frac{z - x_t}{1-t} = \frac{(1-t)(z-\epsilon)}{1-t} = z - \epsilon
$$

The $t$ cancels — the velocity is **constant**. Straight-line path = constant direction. This is why RF is fast: an Euler solver with one step is exact for straight lines.

---

## Reparameterization: What Should the Model Predict?

Differentiating gives $\dot x_t = \dot\alpha_t z + \dot\beta_t \epsilon$, with two unknowns constrained by $x_t$. Knowing $x_t$ and $t$, specifying either $z$ or $\epsilon$ determines the other — so there is one degree of freedom, and the model can predict any one of:

| Parameterization | Predicts | Recover $u^*$ (RF) |
|---|---|---|
| Velocity | $u^* = z - \epsilon$ | use directly |
| Data ($x_0$-pred) | $\hat z$ | $u^* = (\hat z - x_t)/(1-t)$ |
| Noise ($\epsilon$-pred) | $\hat\epsilon$ | $u^* = (x_t - \hat\epsilon)/t$ |

All three are equivalent at the optimum, but imply different **loss weighting** over $t$. For RF, switching to noise prediction scales the per-timestep gradient by $1/t^2$, up-weighting high noise. DDPM favored $\epsilon$-prediction for this reason; RF naturally uses velocity since $z - \epsilon$ is constant.

> [!question] Why doesn't the model take $\epsilon$ as input?
> At training you know $(z, \epsilon, x_t)$ — all three. If the model saw $\epsilon$, it could recover $z = (x_t - (1-t)\epsilon)/t$ algebraically — nothing about $p_\text{data}$ would be learned.
>
> At inference there is no "true" pair: you sample $x_0 \sim \mathcal{N}(0,I)$ and run the ODE. After the first step, $x_h$ is slightly off any straight-line path due to finite step size — feeding $\epsilon = x_0$ would give a spurious $z$ estimate. The model instead learns $u_t(x_t) = \mathbb{E}[z-\epsilon \mid x_t]$, the average over all consistent pairs, which self-corrects under numerical drift. Not seeing $\epsilon$ is what forces the model to learn the data manifold.

---

### What's really going on training and inference

So, what one would do is generate training data. We generate it in a supervised learning way, which is basically sampling here, because you just need to sample:
1. Your z
2. Your time step
3. Your noise (Gaussian noise or something)
4. Calculate $x$ based on these. E.g. $x = tz + (1 -t)\epsilon$.

We just need to make sure the probability path we choose converge to $z$ in the end and is $x_0$ in the beginning.


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
