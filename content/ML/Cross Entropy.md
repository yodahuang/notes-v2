---
date: 2026-02-07
---

*Generated via Claude 4.6 Opus, resulted from a conversation.*

---

Cross-entropy loss is the standard loss function for classification. It can be derived from multiple perspectives — maximum likelihood estimation, information theory, and KL divergence — all of which converge on the same formula. This page walks through the derivation and then consolidates the views.

## Derivation from Maximum Likelihood

Assuming i.i.d. data, the likelihood of the dataset is $L(\theta) = \prod_i P(y_i \mid x_i; \theta)$. Taking the log converts the product to a sum, and flipping the sign (so we can minimize) gives the **Negative Log-Likelihood (NLL)**:

$$
J(\theta) = -\sum_{i=1}^{N} \log P(y_i \mid x_i; \theta)
$$

For a single sample with true class $c$, this is simply $-\log \hat{y}_c$.

> [!note] No $p(k)$ here
> Notice that this derivation never introduces a weighting by $p(k)$. It only says: for each sample, penalize $-\log$ of the predicted probability **at the correct class**. The connection to the full cross-entropy formula $-\sum_k p(k) \log q(k)$ only becomes clear when we aggregate samples — see [[#Why These Views Agree|below]].

---

## Cross-Entropy from Information Theory

For two distributions $p$ (truth) and $q$ (prediction), cross-entropy is defined as:

$$
H(p, q) = -\sum_{k} p(k) \log q(k)
$$

The $p(k)$ weighting comes from taking an **expectation under the true distribution**. The quantity $-\log q(k)$ is the "surprise" or information content of seeing event $k$ under model $q$. Cross-entropy is the expected surprise when reality follows $p$ but you're using $q$:

$$
H(p, q) = \mathbb{E}_{k \sim p}[-\log q(k)]
$$

> [!tip] Why weight by $p$ and not $q$?
> Because we're asking: "how costly is it to use code $q$ when reality is $p$?" In coding theory, $-\log q(k)$ is the code length assigned to event $k$, and $p(k)$ is how often that event actually occurs. The expected message length under reality is what matters.

### Collapse to NLL in Classification

In supervised classification, the true label is one-hot: $p(c) = 1$ for the correct class, $p(k) = 0$ otherwise. The sum collapses:

$$
H(p, q) = -(1) \cdot \log q(c) - \sum_{k \neq c} (0) \cdot \log q(k) = -\log q(c)
$$

This is exactly the per-sample NLL from the likelihood derivation.

---

## Why These Views Agree

The agreement is not a coincidence. All three framings answer the same question: **how well does $q$ approximate $p$?**

### The $p(k)$ weighting emerges from data

In the empirical setting, we don't know $p$ analytically — we have $N$ samples. The empirical distribution is $\hat{p}(k) = n_k / N$ where $n_k$ counts class $k$. The average NLL naturally becomes:

$$
-\frac{1}{N}\sum_{i=1}^{N} \log q(y_i) = -\sum_k \frac{n_k}{N} \log q(k) = -\sum_k \hat{p}(k) \log q(k) = H(\hat{p}, q)
$$

The $p(k)$ weighting is not an axiom introduced from information theory — it **falls out of aggregating per-sample log-likelihoods by class frequency**.

### KL divergence as the unifying frame

Both views are special cases of minimizing the [[KL Divergence]] between the true and predicted distributions:

$$
D_{KL}(p | q) = H(p, q) - H(p)
$$

Since $H(p)$ is constant w.r.t. model parameters, minimizing KL divergence $\Leftrightarrow$ minimizing cross-entropy $\Leftrightarrow$ minimizing NLL.

### Summary

| Perspective        | What you're doing                           | Formula (single sample)                    |
| ------------------ | ------------------------------------------- | ------------------------------------------ |
| Maximum Likelihood | Maximize probability of observed data       | $-\log q(c)$                               |
| Information Theory | Minimize expected surprise under true dist. | $-\sum_k p(k) \log q(k)$                   |
| KL Divergence      | Minimize divergence from true to predicted  | $H(p,q) - H(p)$, constant shift from above |

For one-hot labels, all three reduce to $-\log q(c)$.

---

## See Also

- [[KL Divergence]]
- [[Fisher Information]]
- [[Softmax Function]]