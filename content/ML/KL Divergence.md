---
date: 2026-02-07
---
*Generated via Claude 4.6 Opus, resulted from a conversation.*

---

The Kullback-Leibler divergence measures how one probability distribution $q$ differs from a reference distribution $p$:

$$
D_{KL}(p | q) = \sum_k p(k) \log \frac{p(k)}{q(k)} = -\sum_k p(k) \log q(k) + \sum_k p(k) \log p(k) = H(p, q) - H(p)
$$

It has several independent derivations. The most concrete starts from coding theory.

---

## Coding Theory Motivation

> [!question] Why start from coding theory for a quantity about probability distributions?
> A probability distribution assigns beliefs over outcomes. The coding question asks: if you had to _act_ on those beliefs — commit resources under uncertainty — how costly would it be? Code length is the most stripped-down version of that: allocating a finite resource (bits) based on how likely you think each outcome is.
> 
> $-\log p(k)$ keeps appearing not because we care about bits per se, but because the logarithm is the **unique function** that converts multiplicative probability structure into additive cost structure while respecting independence. Coding theory is just the cleanest context where that becomes concrete — you can point at a literal string of bits and say "this is what your wrong beliefs cost you."
> 
> You could skip coding entirely: define $D_{KL}$ by formula, prove non-negativity, show uniqueness from axioms. That's mathematically complete but unmotivated. The coding interpretation gives physical intuition for why each piece is there — why $p(k)$ weights, why $\log$, why the ratio $p/q$.

### The Setup: Encoding Messages

Suppose we need to transmit a sequence of symbols (events, classes, outcomes) over a channel. Each symbol is drawn from some distribution $p$. We want to assign a binary code to each symbol — shorter codes for frequent symbols, longer codes for rare ones.

For example, if we have four symbols with equal probability $1/4$, we need 2 bits each: `00`, `01`, `10`, `11`. But if symbol A occurs 50% of the time, we'd prefer to give it a shorter code.

### The Kraft Inequality: What Constrains Code Lengths

For a code to be uniquely decodable (no codeword is a prefix of another), the code lengths $\ell_k$ must satisfy:

$$
\sum_k 2^{-\ell_k} \leq 1
$$

This is the **Kraft inequality**. It's a hard constraint from the structure of binary prefix codes — it limits how many short codewords you can have. If you make one code shorter, others must get longer.

### Optimal Code Lengths → Entropy

Given this constraint, what code lengths minimize the expected cost $\sum_k p(k), \ell_k$? This is a constrained optimization problem. Using Lagrange multipliers, the solution is:

$$
\ell_k^* = -\log p(k)
$$

Plugging back in, the minimum achievable expected code length is:

$$
\sum_k p(k), \ell_k^* = -\sum_k p(k) \log p(k) = H(p)
$$

This is **entropy** — not defined axiomatically, but _derived_ as the solution to "minimize expected code length subject to Kraft." That's Shannon's source coding theorem: no lossless code beats $H(p)$, and codes exist that come arbitrarily close.

> [!note] 
> Fractional bits $-\log p(k)$ is generally not an integer. Practical codes like Huffman coding round to integers (achieving average length between $H(p)$ and $H(p)+1$). Arithmetic coding gets arbitrarily close to $H(p)$ by encoding sequences jointly rather than symbol-by-symbol. The theorem is an asymptotic statement about what's achievable in principle.

### Using the Wrong Code → Cross-Entropy

Now suppose reality follows $p$, but we design our code for distribution $q$. By the same Kraft argument, the optimal code lengths for $q$ are $-\log q(k)$. But symbols still occur with frequency $p(k)$. The expected code length becomes:

$$
\sum_k p(k)(-\log q(k)) = H(p, q)
$$

This is **cross-entropy**. The $p(k)$ weighting doesn't change — reality is still $p$ — we've just plugged in the wrong code lengths.

### The Excess Cost → KL Divergence

The difference between what we pay and what we could have paid:

$$
H(p, q) - H(p) = \sum_k p(k) \log \frac{p(k)}{q(k)} = D_{KL}(p | q)
$$

KL divergence is the **extra bits per symbol** you pay for using $q$ instead of $p$.

> [!note] "Surprise" is an interpretation, not the foundation 
> $-\log p(k)$ is often introduced as the "surprise" of event $k$, with entropy defined as "expected surprise." This is a useful interpretation — the quantity is indeed zero for certain events, large for rare ones, and additive for independent events. But these properties aren't axioms chosen because "surprise should work this way." They're consequences of $-\log p(k)$ being the optimal code length under the Kraft inequality. "Surprise" is a name we give to the result after the fact.

---

## Other Derivations

### Hypothesis Testing

Given $N$ i.i.d. samples from $p$, the expected log-likelihood ratio between $p$ and $q$ is:

$$
\frac{1}{N}\sum_{i=1}^N \log \frac{p(x_i)}{q(x_i)} \xrightarrow{N \to \infty} D_{KL}(p | q)
$$

KL divergence is the **expected evidence per sample** in favor of $p$ over $q$ when $p$ is true. This connects to Stein's lemma: $D_{KL}$ governs the exponential decay rate of Type II error in hypothesis testing.

### Axiomatic Uniqueness

KL divergence can be characterized as essentially the unique divergence satisfying:

- $D(p | q) \geq 0$ with equality iff $p = q$
- Additivity over independent variables
- Invariance under sufficient statistics (the data processing inequality)

This is formalized through Csiszár's work on $f$-divergences.

---

## Connection to Riemannian Geometry

For two nearby distributions $p_\theta$ and $p_{\theta + d\theta}$ on the [[Statistical Manifold]], KL divergence to second order is:

$$
D_{KL}(p_\theta | p_{\theta + d\theta}) \approx \frac{1}{2}d\theta^T F(\theta)d\theta
$$

where $F(\theta)$ is the [[Fisher Information]] matrix. The Fisher metric is the **local quadratic approximation** of KL divergence — this is how information geometry derives the Riemannian structure of probability space from KL divergence, not the other way around.

This is why the [[Natural Policy Gradient|Natural Gradient]] — which uses $F(\theta)^{-1}\nabla J$ — is the steepest descent direction in the geometry induced by KL divergence rather than Euclidean distance.

---

## Properties

- **Non-negative:** $D_{KL}(p | q) \geq 0$ (Gibbs' inequality), with equality iff $p = q$
- **Asymmetric:** $D_{KL}(p | q) \neq D_{KL}(q | p)$ in general — it is not a true metric
- **Not a distance:** violates triangle inequality and symmetry
- **Additive:** for independent variables, $D_{KL}(p_1 p_2 | q_1 q_2) = D_{KL}(p_1 | q_1) + D_{KL}(p_2 | q_2)$

---

## See Also

- [[Cross Entropy]]
- [[Fisher Information]]
- [[Natural Policy Gradient|Natural Gradient]]
- [[Statistical Manifold]]