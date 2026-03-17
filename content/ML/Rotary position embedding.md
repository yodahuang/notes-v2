---
aliases:
  - RoPE
Arxiv: https://arxiv.org/abs/2104.09864
pdf: "[[roformer.pdf]]"
original title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
date: 2025-09-14
tags:
year: 2021
updated: 2026-03-15
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

$$
a_{m,n}=\frac{\exp(\frac{\boldsymbol{q}_m^\intercal\boldsymbol{k}_n}{\sqrt{d}})}{\sum_{j=1}^N\exp(\frac{\boldsymbol{q}_m^\intercal\boldsymbol{k}_j}{\sqrt{d}})}
$$

For the traditional absolute position embedding, that's

$$
f_{t:t\in\{q,k,v\}}(\boldsymbol{x}_i,i):=\boldsymbol{W}_{t:t\in\{q,k,v\}}(\boldsymbol{x}_i+\boldsymbol{p}_i),
$$

Since we want to capture the relative information and attention, it would be good if that $q^T_mk_n$ depend only on the relative position between $m$ and $n$. So the problem becomes: can we find such an $f$, such that

$$
\langle f_q(\boldsymbol{x}_m,m),f_k(\boldsymbol{x}_n,n)\rangle=g(\boldsymbol{x}_m,\boldsymbol{x}_n,m-n).
$$

One can find a solution to our formulation, when $d=2$ is:

$$
\begin{aligned}f_{q}(\boldsymbol{x}_{m},m)&=(\boldsymbol{W}_q\boldsymbol{x}_m)e^{im\theta}\\f_k(\boldsymbol{x}_n,n)&=(\boldsymbol{W}_k\boldsymbol{x}_n)e^{in\theta}\\g(\boldsymbol{x}_m,\boldsymbol{x}_n,m-n)&=\mathrm{Re}[(\boldsymbol{W}_q\boldsymbol{x}_m)(\boldsymbol{W}_k\boldsymbol{x}_n)^*e^{i(m-n)\theta}]\end{aligned}
$$

where $\mathrm{Re}[\cdot]$ is the real part of a complex number and $(\boldsymbol{W}_k\boldsymbol{x}_n)^*$ represents the conjugate complex number of $(\boldsymbol{W}_k\boldsymbol{x}_n)$. $\theta\in\mathbb{R}$ is a preset non-zero constant. We can further write $f_\{q,k\}$ in a multiplication matrix:

$$
f_{\{q,k\}}(\boldsymbol{x}_m,m)=\left(\begin{array}{cc}\cos m\theta&-\sin m\theta\\\sin m\theta&\cos m\theta\end{array}\right)\left(\begin{array}{cc}W_{\{q,k\}}^{(11)}&W_{\{q,k\}}^{(12)}\\W_{\{q,k\}}^{(21)}&W_{\{q,k\}}^{(22)}\end{array}\right)\left(\begin{array}{c}x_m^{(1)}\\x_m^{(2)}\end{array}\right)
$$

We can see intuitively that this work since this is rotating the embedding, or, assigning them angles in 2D plane. A vector at $\theta$ and a vector at $\gamma$, when doing dot product, gives us $\cos(\theta - \gamma)$.

In order to generalize our results in 2D to any $x_i \in \mathbb{R}^d$ where $d$ is even, we divide the d-dimension space into $d/2$ sub-spaces and combine them in the merit of the [[#Appendix A linearity of the inner product|linearity of the inner product]], turning $f_{\{q,k\}}$ into:

$$
f_{\{q,k\}}(x_m, m) = R^d_{\Theta, m} W_{\{q,k\}} x_m
$$

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

- $\theta$ mutations are the same idea as [[Absolute position embedding]]. It's for differentiating dimensions. They can technically be the same and still satisfies RoPE property. They are still set as this since:
	- You want frequencies spread across many orders of magnitude so different heads/dimensions can attend to both local and long-range positional relationships.
	- You want the rotation to be slow enough in some dimensions that the model can generalize to longer sequences.
	- The predefined θ sequence is more of a **prior / initialization of the frequency basis** than a hard constraint. The model has a degree of freedom to work around it through the learned projections.
- $d$ mutations is on position.
## Comparison

|                     | Absolute PE                               | RoPE                                                             |
| ------------------- | ----------------------------------------- | ---------------------------------------------------------------- |
| **What it encodes** | Absolute position                         | Relative position                                                |
| **Where applied**   | Input embeddings                          | Q and K, inside attention                                        |
| **Why there**       | Enriches token identity before projection | Must survive to the dot product to get `R_{j-i}` factoring       |
| **Applied to V?**   | Yes (implicitly, V gets the embedding)    | No — V doesn't participate in the position-sensitive dot product |

## Properties of RoPE
- Similar as the OG position embedding, it has a long term decay property (that's because only the "going down" part of the cosine is the dominant part)
- It can be used easily with linear attention

## Implementation

This part is generated from my conversation with Claude Sonnet 4.6.

Rather than constructing the full block-diagonal rotation matrix, we exploit the fact that each 2D rotation only mixes adjacent pairs $(x_{2i}, x_{2i+1})$. This lets us decompose the operation into two elementwise multiplications.
#### Derivation (d=6)

For $x = [x_0, x_1, x_2, x_3, x_4, x_5]$ with frequency angles $\theta_0, \theta_1, \theta_2$, the full rotation gives:

$$
x' = \begin{bmatrix} 
x_0 \cos\theta_0 - x_1 \sin\theta_0 \\ 
x_1 \cos\theta_0 + x_0 \sin\theta_0 \\ 
x_2 \cos\theta_1 - x_3 \sin\theta_1 \\ 
x_3 \cos\theta_1 + x_2 \sin\theta_1 \\ 
x_4 \cos\theta_2 - x_5 \sin\theta_2 \\ 
x_5 \cos\theta_2 + x_4 \sin\theta_2 
\end{bmatrix}
$$

Split into two terms:

$$
x' = \underbrace{
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \end{bmatrix} \odot 
\begin{bmatrix} \cos\theta_0 \\ \cos\theta_0 \\ \cos\theta_1 \\ \cos\theta_1 \\ \cos\theta_2 \\ \cos\theta_2 \end{bmatrix}
}_{\text{term 1}} + 
\underbrace{
\begin{bmatrix} -x_1 \\ x_0 \\ -x_3 \\ x_2 \\ -x_5 \\ x_4 \end{bmatrix} \odot 
\begin{bmatrix} \sin\theta_0 \\ \sin\theta_0 \\ \sin\theta_1 \\ \sin\theta_1 \\ \sin\theta_2 \\ \sin\theta_2 \end{bmatrix}
}_{\text{term 2}}
$$

> [!note] Why this split and not another?
> 
> We could have split differently — e.g. putting only even-indexed elements in term 1 and odd-indexed in term 2. But that would require **strided slicing** and **interleaving** back at the end, which are awkward tensor operations.
> 
> This split is chosen because both terms have the same "shape of access" — the full vector $x$ appears once in each term. That means term 2 only requires a cheap rearrangement of $x$, not a gather/scatter.
> 
> **Principle:** the math admits many equivalent decompositions. Choose the one that maps onto cheap tensor operations.

---

## Computing the Two Components

The cos/sin vectors go from shape `[seq_len, d//2]` → `[seq_len, d]` via `repeat_interleave(..., repeats=2, dim=-1)`.

The `rotate_half` operation transforms $x$ into $[-x_1, x_0, -x_3, x_2, -x_5, x_4]$:

$$
\begin{bmatrix} x_0 & x_1 \\ x_2 & x_3 \\ x_4 & x_5 \end{bmatrix} 
\xrightarrow{\text{flip}(-1)} 
\begin{bmatrix} x_1 & x_0 \\ x_3 & x_2 \\ x_5 & x_4 \end{bmatrix} 
\xrightarrow{\times [-1, 1]} 
\begin{bmatrix} -x_1 & x_0 \\ -x_3 & x_2 \\ -x_5 & x_4 \end{bmatrix} 
\xrightarrow{\text{flatten}} 
\begin{bmatrix} -x_1 \\ x_0 \\ -x_3 \\ x_2 \\ -x_5 \\ x_4 \end{bmatrix}
$$


---

## Final Formula

$$
x' = x \odot \cos\theta + \operatorname{rotate\_half}(x) \odot \sin\theta
$$

---

## Intuitions

> [!tip] The "passengers" intuition
> 
> The rotation only acts on the $d$ dimension. Batch, heads, and seq are just passengers — they don't participate in the logic. So you can derive everything by thinking about a single vector $x \in \mathbb{R}^d$, then broadcast freely over the other dimensions.
> 
> A tensor of shape `[seq_len, d]` for cos/sin already holds the answer for _every_ position simultaneously. There is no loop — broadcasting _is_ the loop.

> [!tip] Tensor thinking: work backwards from target shape
> 
> The key question when vectorizing is: _"what arrangement of elements would make this a simple elementwise operation?"_
> 
> For `rotate_half`, we wanted $[-x_1, x_0, \ldots]$. Working backwards: that's a flatten of a `[d//2, 2]` matrix where columns are swapped and column 0 is negated. Swap → `flip`. Negate column 0 → multiply by `[-1, 1]`. The loop is never written; it dissolves into shape manipulation.


My own implementation:
```python
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        # In the original paper they use m in place of pos,and i in place of k.
        pos_s = torch.arange(max_seq_len)
        k_half_s = torch.arange(d_k // 2)

        theta_p_k: Num[Tensor, "seq_len d_k_half"] = torch.outer(pos_s, (theta ** (-2 * k_half_s / d_k)))
        cos_theta_p_k: Num[Tensor, "seq_len d_k_half"] = torch.cos(theta_p_k)
        sin_theta_p_k: Num[Tensor, "seq_len d_k_half"] = torch.sin(theta_p_k)

        # seq_len dimensions are passers. The important stuff is the k dim.
        # Take an example of last_dim=6
        # For each pair we do ratation
        # Reorder them so there's no interleaving
        term_one: Num[Tensor, "seq_len d_k"] = torch.repeat_interleave(cos_theta_p_k, repeats=2, dim=-1)
        term_two: Num[Tensor, "seq_len d_k"] = torch.repeat_interleave(sin_theta_p_k, repeats=2, dim=-1)
        self.register_buffer("term_one", term_one, persistent=False)
        self.register_buffer("term_two", term_two, persistent=False)

    def forward(
        self, x: Num[Tensor, "... seq_len d_k"], token_positions: Integer[Tensor, "... seq_len"]
    ) -> Num[Tensor, "... seq_len d_k"]:
        # For non packed data that token_positions is an arange in training time
        # x1 part is easy. focusing on x2...
        xx: Num[Tensor, "... d_k_half 2"] = einx.rearrange("... (d_k_half two) -> ... d_k_half two", x, two=2)
        xx = xx * torch.tensor([1, -1], device=self.term_one.device)
        xx = xx.flip([-1])
        # Can also just do this, may be more intuitive
        # xx = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1)
        xx = einx.rearrange("... d_k_half two -> ... (d_k_half two)", xx, two=2)
        return self.term_one[token_positions] * x + self.term_two[token_positions] * xx

```

## Appendix A: linearity of the inner product

Let's consider two vectors, $\mathbf{x}$ and $\mathbf{y}$, in a $d$-dimensional space, where $d$ is an even number.

$$
\mathbf{x} = (x_1, x_2, x_3, x_4, \dots, x_{d-1}, x_d)
$$

$$
\mathbf{y} = (y_1, y_2, y_3, y_4, \dots, y_{d-1}, y_d)
$$

The standard inner product (dot product), denoted by $\langle \mathbf{x}, \mathbf{y} \rangle$, is defined as the sum of the element-wise products of their components:

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{d} x_i y_i = x_1y_1 + x_2y_2 + x_3y_3 + \dots + x_dy_d
$$

Now, RoPE treats the $d$-dimensional vector as a concatenation of $d/2$ smaller, 2-dimensional vectors. Let's denote these sub-vectors with a prime symbol (′):

* $\mathbf{x'}_1 = (x_1, x_2)$, $\mathbf{x'}_2 = (x_3, x_4)$, ..., $\mathbf{x'}_{d/2} = (x_{d-1}, x_d)$
* $\mathbf{y'}_1 = (y_1, y_2)$, $\mathbf{y'}_2 = (y_3, y_4)$, ..., $\mathbf{y'}_{d/2} = (y_{d-1}, y_d)$

Because of the basic rules of addition, we can simply regroup the terms in the original inner product sum:

$$
\langle \mathbf{x}, \mathbf{y} \rangle = (x_1y_1 + x_2y_2) + (x_3y_3 + x_4y_4) + \dots + (x_{d-1}y_{d-1} + x_dy_d)
$$

Notice that each term in parentheses is just the inner product of the corresponding 2D sub-vectors:

* $(x_1y_1 + x_2y_2) = \langle \mathbf{x'}_1, \mathbf{y'}_1 \rangle$
* $(x_3y_3 + x_4y_4) = \langle \mathbf{x'}_2, \mathbf{y'}_2 \rangle$

This leads us to the core identity that RoPE exploits. The inner product in $d$-dimensions is precisely the sum of the inner products in the $d/2$ constituent 2D subspaces:

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{d/2} \langle \mathbf{x'}_i, \mathbf{y'}_i \rangle
$$