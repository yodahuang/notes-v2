---
date: 2026-04-04
pdf: "[[sovit.pdf]]"
year: 2023
---
The paper basically covers how to do [[Scaling Law]] analysis on a [[Vision Transformer (ViT)|ViT]].

>Our method involves both a functional form (2) and a novel procedure
[[sovit.pdf#page=5&selection=159,26,163,21|sovit, page 5]]

## The functional form

- $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_D) \in \mathbb{N}^D$ containing $D$ shape dimensions, such as width, depth and MLP size. This is the neural architecture. 
- $t$, compute such as GFLOPS, $\propto 6 \text{Data} * x$.
- $f : \mathbb{N}^D \times \mathbb{R}^+ \to \mathbb{R}$  A performance metric of interest, such as downstream ImageNet 10-shot error rate. Specifically, $f(\mathbf{x}, \mathbf{t})$ results from (pre)-training an architecture $\mathbf{x}$ for a fixed compute budget $\mathbf{t}$. We always assume that $f$ corresponds to a loss, meaning lower values are better.

For each dimension $x_{k}$, the paper argues that the function form is as follows:
$$\begin{aligned} f_k(\mathbf{x}_k, \mathbf{t}) \sim \alpha_k \mathbf{x}_k^{-a_k} + (\beta_k \mathbf{x}_k^{b_k} + \xi_k) \mathbf{t}^{-c} + \varepsilon_k, \end{aligned}$$
where $\alpha_k, a_k, \beta_k, b_k, c, \xi_k, \varepsilon_k > 0$. Here, $f_k$ focuses on the dimension $k$ alone and assumes that all other shape dimensions $j \neq k$ are sufficiently large such that they do not constitute a bottleneck.

Why? 

> Our argument for this particular functional form is six-fold

[[sovit.pdf#page=4&selection=306,0,306,60|sovit, page 4]]

And I'll omit that in this note.

Now let's take the derivative and set to zero w.r.t. $x_k$: we got the optimal one:
$$\boxed{\mathbf{x}_k^\star = \left(\frac{\alpha_k a_k \mathbf{t}^c}{\beta_k b_k}\right)^{\frac{1}{b_k + a_k}}}$$

Thus $\mathbf{x}_k^\star \propto \mathbf{t}^{c/(b_k + a_k)}$, and we call that exponent $s_k$

We can then derive that
$$f_k(\mathbf{x}_k^\star, t) = F(\mathbf{x}_k^\star)^{-a_k} + Gt^{-c} + \varepsilon_k$$

## The procedure

### Star Sweep

Start from a *large* and *random* model, use that as the star center, and then varying a single dimension $k\in [D]$ at a time in an exponentially-spaced-grid, going **down**. In practice, they use width, depth and MLP dim. They only go down to make sure other dims do not form a bottleneck when estimating the params.

For example, their start center is $(1968, 40, 6144)$. For the MLP center, they used grid $(1088, 1360, 1728, 2160, 2592, 3072)$, 20% increase in each step. 

This process gives the scaling component, the $s_k$ part.

### Grid Sweep

Now we use *small* models. Grid search! Go test a bunch and find the configuration to be at [[Pareto Front]]. For example, they find it to be $(608, 10, 928)$. This is the leading coefficient of $\mathbf{x}_k^\star$.

### Together

Now we have the scaling factor for *each* dim, and where o start. We can simply scale. In this paper, they simply scale it equally within the budget.