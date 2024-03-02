---
chapter: ??
share: true
---
## Introduction
Optimization problems can be written as follows:
$$
\min_{x\in\mathbb{R}^n} f(x)\qquad \text{subject to }
\begin{cases}
c_i(x) = 0, \quad i \in \mathcal{E}, \\
c_{i}(x) \ge 0, \quad i \in \mathcal{I},
\end{cases}
$$
In the book (Numerical Optimization by Jorge Nocedal & Stephen J. Wright), we focus on continuous local deterministic optimization, and go over both constrained and unconstrained optimization.

## Fundamentals of Unconstrained optimization
We only consider the problem of
$$
\min_{x}f(x)
$$
here.

### Talor's Theorem
This is not only Talor's theorem. We also uses Lagrange theorem (mean value theorem) here.

Suppose that $f : \mathbb{R}^n \rightarrow \mathbb{R}$ is continuously differentiable and that $p\in\mathbb{R}^n$ . Then we have that
$$
\begin{aligned}
f(x+p) &= f(x) + \nabla f(x+tp)^T p \\
&= f(x) + \nabla f(x)^Tp + \frac{1}{2}p^T\nabla^2f(x+tp)p
\end{aligned}
$$
### FIrst Order Necessary Conditions
If $x^∗$ is a local minimizer and f is continuously differentiable in an open neighborhood of $x^∗$ , then $\nabla f (x^∗) = 0$.

### Second Order Necessary Conditions
If $x^∗$ is a local minimizer of f and $\nabla^2f$ exists and is continuous in an open neighborhood of $x^∗$ , then $\nabla f (x^∗ ) = 0$ and $\nabla^2 f (x^∗ )$ is positive semideﬁnite.
### Second-Order Sufﬁcient Conditions
Suppose that $\nabla^2f$  is continuous in an open neighborhood of $x^∗$ and that $\nabla f (x^∗) = 0$ and $\nabla f (x^∗)$ is positive deﬁnite. Then $x^∗$ is a strict local minimizer of $f$ .
