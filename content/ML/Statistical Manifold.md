---
date: 2026-02-08
---

*Generated via Claude 4.6 Opus, resulted from a conversation.*

---

A **statistical manifold** is the space of all probability distributions in a parametric family $\{p_\theta : \theta \in \Theta\}$, treated as a Riemannian manifold. Each point on the manifold is a distribution; the parameters $\theta$ serve as coordinates.

This is the setting of **information geometry**, a branch of differential geometry applied to probability and statistics.

---

## Manifold Structure

A manifold is a space that locally looks like $\mathbb{R}^n$ but may have global curvature. The statistical manifold inherits this structure from its parameterization: for a $d$-parameter family, it is (locally) a $d$-dimensional manifold.

The key point is that **parameter space $\mathbb{R}^d$ is not the manifold itself** — it is a coordinate chart. Different parameterizations (e.g., $(\mu, \sigma)$ vs. $(\mu, \sigma^2)$ vs. the natural parameters of the exponential family) give different charts for the same manifold. The geometric properties — distances, curvature, geodesics — should not depend on which chart you use.

---

## Riemannian Metric: Fisher Information

The [[Fisher Information]] matrix $F(\theta)$ serves as the Riemannian metric tensor. It defines the inner product on the tangent space at each point:

$$
\langle u, v \rangle_{p_\theta} = u^T F(\theta) v
$$

where $u, v$ are tangent vectors. The tangent space is spanned by the [[Score Function]] components $\partial_{\theta_i} \log p_\theta(x)$.

By Čencov's theorem, the Fisher metric is the *unique* (up to scale) Riemannian metric that is invariant under sufficient statistics — i.e., it does not depend on how you choose to represent the data, only on the statistical information it carries.

The [[KL Divergence]] is the divergence function whose local quadratic form coincides with this metric: $D_{KL}(p_\theta \| p_{\theta+d\theta}) \approx \frac{1}{2} d\theta^T F(\theta) d\theta$.

---

## Analogy with Lie Groups

There is a structural parallel (not an equivalence) with Lie groups from robotics:

| Lie groups (e.g., SE(3)) | Statistical manifold |
|---|---|
| Group elements (rigid transforms) | Probability distributions |
| Lie algebra $\mathfrak{se}(3)$ (tangent space at identity) | Tangent space at $p_\theta$ (spanned by score functions) |
| Exponential map $\exp: \mathfrak{g} \to G$ | Exponential family: $p_\theta \propto \exp(\theta^T T(x))$ |
| Left-invariant metric (if defined) | Fisher metric (invariant under sufficient statistics) |

The analogy is conceptual: both settings involve doing calculus on curved spaces where the "natural" geometry differs from Euclidean. In Lie groups, the group structure dictates the geometry; in statistical manifolds, the normalization constraint and information content do.

The word "exponential" appears in both contexts but for different structural reasons — the matrix exponential integrates infinitesimal generators on a Lie group, while the exponential in exponential families ensures positivity and connects to the log-linear structure that makes the [[Score Function]] so clean.

---

## Further Reading

This is a topic in differential geometry. Key references:

- **Amari & Nagaoka**, *Methods of Information Geometry* (2000) — the foundational text on information geometry
- **Sola et al.**, *A micro Lie theory for state estimation in robotics* (2018) — for the Lie group side of the analogy
- **Ay, Jost, Lê, Schwachhöfer**, *Information Geometry* (2017) — modern mathematical treatment
