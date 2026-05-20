---
date: 2026-03-29
---
This is based my conversation with Claude Sonnet 4.6

---
## The problem

We want to draw samples from a target distribution $p(x)$, but cannot sample from it directly. The normalizing constant of $p$ may also be unknown.

## Algorithm

Choose a proposal distribution $q(x)$ you can sample from, and find a scalar $M$ such that:

$$
M \cdot q(x) \geq p(x) \quad \forall x
$$

Then repeat until acceptance:

1. Draw candidate $x \sim q(x)$
2. Draw $u \sim \text{Uniform}(0, 1)$
3. **Accept** $x$ if $u < \dfrac{p(x)}{M \cdot q(x)}$, else reject

Accepted samples are distributed exactly according to $p$.

> [!tip] Unnormalized densities work
> Only $\tilde{p}(x) \propto p(x)$ is needed — the unknown normalizing constant cancels in the acceptance ratio. This is one of the main practical motivations for rejection sampling.

## Intuition

Think of $M \cdot q(x)$ as a ceiling above $p(x)$. Sampling a candidate $x$ picks a random point under the ceiling at that location. The acceptance coin is weighted by how much of that vertical bar lies below $p(x)$, so $x$ is kept with probability proportional to $p(x)$.

![[rejection_sampling.svg]]

## The catch — curse of dimensionality

The acceptance rate is $1/M$. In high dimensions, even a slightly mismatched proposal causes $M$ to grow exponentially: if acceptance rate per dimension is 90%, across 100 dimensions it is $0.9^{100} \approx 0.003%$. Rejection sampling is practical only in low dimensions.

$M$ remains a **scalar** in the multivariate case — it is one global constant that must cover the entire target over $\mathbb{R}^d$.

## Comparison with [[Importance sampling corrections]]

Both methods use the ratio $p(x)/q(x)$, but answer different questions.

|                  | Rejection sampling         | Importance sampling                    |
| ---------------- | -------------------------- | -------------------------------------- |
| Output           | Exact iid samples from $p$ | Weighted estimate of $\mathbb{E}_p[f]$ |
| Requires $M$?    | Yes                        | No                                     |
| Accepts/rejects? | Yes                        | No                                     |
| High dimensions  | Breaks down (rate → 0)     | Degrades gracefully                    |
| Failure mode     | Exponential rejection      | Variance blows up                      |

Importance sampling rewrites the expectation as:

$$
\mathbb{E}_p[f(x)] = \mathbb{E}_q\left[f(x) \cdot \frac{p(x)}{q(x)}\right]
$$

so you draw $x_i \sim q$ and average $f(x_i) \cdot w(x_i)$ with weights $w(x_i) = p(x_i)/q(x_i)$. No rejection, no $M$. Its failure mode is variance blow-up when $q$ has lighter tails than $p$, causing a few samples to dominate with huge weights.

Importance sampling is far more common in ML (variational inference, policy gradients, off-policy RL) due to better high-dimensional scaling.
