---
aliases:
  - RoPE
Arxiv: https://arxiv.org/abs/2104.09864
pdf: "[[roformer.pdf]]"
original title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
date: 2025-09-14
tags:
year: 2021
---
The paper was published in 2021 as a preliminary report, and in late 2023 it has been updated to v5. The core idea is the same. The only thing that's changed is basically evaluating on English benchmarks instead of Chinese data only. 

This is probably one of the papers with the worst English writing skills I've read. It's kinda iconic since this is an LLM paper, but LLM is not being used to audit the writing of the paper.

## Why relative
This is my understanding, not what's in the paper. We don't need the absolute position information that's done in [[Absolute position embedding]]. What we care about is the relative distance when doing attention. That means the y can generalize better on longer sequence (position 5 to 8 is the same as 500 to 503).

## Formulation
Say we generate the positional embedding via function $f$, 
$$
\begin{aligned}
q_m &= f_q(\mathbf{x}_m, m) \\
\mathbf{k}_n &= f_k(\mathbf{x}_n, n) \\
\mathbf{v}_n &= f_v(\mathbf{x}_n, n),
\end{aligned}
$$
$k$ and $v$ are both on $n$ position because that's a way of understanding the weighted sum.
And then they go through softmax like this

$$a_{m,n}=\frac{\exp(\frac{\boldsymbol{q}_m^\intercal\boldsymbol{k}_n}{\sqrt{d}})}{\sum_{j=1}^N\exp(\frac{\boldsymbol{q}_m^\intercal\boldsymbol{k}_j}{\sqrt{d}})}$$

For the traditional absolute position embedding, that's

$$f_{t:t\in\{q,k,v\}}(\boldsymbol{x}_i,i):=\boldsymbol{W}_{t:t\in\{q,k,v\}}(\boldsymbol{x}_i+\boldsymbol{p}_i),$$

Since we want to capture the relative information and attention, it would be good if that $q^T_mk_n$ depend only on the relative position between $m$ and $n$. So the problem becomes: can we find such an $f$, such that

$$\langle f_q(\boldsymbol{x}_m,m),f_k(\boldsymbol{x}_n,n)\rangle=g(\boldsymbol{x}_m,\boldsymbol{x}_n,m-n).$$

One can find a solution to our formulation, when $d=2$ is:

$$\begin{aligned}f_{q}(\boldsymbol{x}_{m},m)&=(\boldsymbol{W}_q\boldsymbol{x}_m)e^{im\theta}\\f_k(\boldsymbol{x}_n,n)&=(\boldsymbol{W}_k\boldsymbol{x}_n)e^{in\theta}\\g(\boldsymbol{x}_m,\boldsymbol{x}_n,m-n)&=\mathrm{Re}[(\boldsymbol{W}_q\boldsymbol{x}_m)(\boldsymbol{W}_k\boldsymbol{x}_n)^*e^{i(m-n)\theta}]\end{aligned}$$

where $\mathrm{Re}[\cdot]$ is the real part of a complex number and $(\boldsymbol{W}_k\boldsymbol{x}_n)^*$ represents the conjugate complex number of $(\boldsymbol{W}_k\boldsymbol{x}_n)$. $\theta\in\mathbb{R}$ is a preset non-zero constant. We can further write $f_\{q,k\}$ in a multiplication matrix:

$$f_{\{q,k\}}(\boldsymbol{x}_m,m)=\left(\begin{array}{cc}\cos m\theta&-\sin m\theta\\\sin m\theta&\cos m\theta\end{array}\right)\left(\begin{array}{cc}W_{\{q,k\}}^{(11)}&W_{\{q,k\}}^{(12)}\\W_{\{q,k\}}^{(21)}&W_{\{q,k\}}^{(22)}\end{array}\right)\left(\begin{array}{c}x_m^{(1)}\\x_m^{(2)}\end{array}\right)$$

We can see intuitively that this work since this is rotating the embedding, or, assigning them angles in 2D plane. A vector at $\theta$ and a vector at $\gamma$, when doing dot product, gives us $\cos(\theta - \gamma)$.

In order to generalize our results in 2D to any $x_i \in \mathbb{R}^d$ where $d$ is even, we divide the d-dimension space into $d/2$ sub-spaces and combine them in the merit of the [[#Appendix A linearity of the inner product|linearity of the inner product]], turning $f_{\{q,k\}}$ into:

$$f_{\{q,k\}}(x_m, m) = R^d_{\Theta, m} W_{\{q,k\}} x_m$$

where

$$
R^d_{\Theta, m} =
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}
$$

is the rotary matrix with pre-defined parameters $\Theta = \{\theta_i = 10000^{-2(i-1)/d}, i \in [1, 2, ..., d/2]\}$.  One can think about it as "encoding relative information for each pair of size 2 in the embedding".

## Properties of RoPE
- Similar as the OG position embedding, it has a long term decay property (that's because only the "going down" part of the cosine is the dominant part)
- It can be used easily with linear attention
## Appendix A: linearity of the inner product

Let's consider two vectors, $\mathbf{x}$ and $\mathbf{y}$, in a $d$-dimensional space, where $d$ is an even number.

$$\mathbf{x} = (x_1, x_2, x_3, x_4, \dots, x_{d-1}, x_d)$$
$$\mathbf{y} = (y_1, y_2, y_3, y_4, \dots, y_{d-1}, y_d)$$

The standard inner product (dot product), denoted by $\langle \mathbf{x}, \mathbf{y} \rangle$, is defined as the sum of the element-wise products of their components:

$$\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{d} x_i y_i = x_1y_1 + x_2y_2 + x_3y_3 + \dots + x_dy_d$$

Now, RoPE treats the $d$-dimensional vector as a concatenation of $d/2$ smaller, 2-dimensional vectors. Let's denote these sub-vectors with a prime symbol (′):

* $\mathbf{x'}_1 = (x_1, x_2)$, $\mathbf{x'}_2 = (x_3, x_4)$, ..., $\mathbf{x'}_{d/2} = (x_{d-1}, x_d)$
* $\mathbf{y'}_1 = (y_1, y_2)$, $\mathbf{y'}_2 = (y_3, y_4)$, ..., $\mathbf{y'}_{d/2} = (y_{d-1}, y_d)$

Because of the basic rules of addition, we can simply regroup the terms in the original inner product sum:

$$\langle \mathbf{x}, \mathbf{y} \rangle = (x_1y_1 + x_2y_2) + (x_3y_3 + x_4y_4) + \dots + (x_{d-1}y_{d-1} + x_dy_d)$$

Notice that each term in parentheses is just the inner product of the corresponding 2D sub-vectors:

* $(x_1y_1 + x_2y_2) = \langle \mathbf{x'}_1, \mathbf{y'}_1 \rangle$
* $(x_3y_3 + x_4y_4) = \langle \mathbf{x'}_2, \mathbf{y'}_2 \rangle$

This leads us to the core identity that RoPE exploits. The inner product in $d$-dimensions is precisely the sum of the inner products in the $d/2$ constituent 2D subspaces:

$$\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{d/2} \langle \mathbf{x'}_i, \mathbf{y'}_i \rangle$$