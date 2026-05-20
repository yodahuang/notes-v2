---
date: 2026-04-06
---

```pseudo
\begin{algorithm}
\caption{Speculative Sampling (SpS) with Auto-Regressive Target and Draft Models}
\begin{algorithmic}
\STATE Given lookahead $K$ and minimum target sequence length $T$.
\STATE Given auto-regressive target model $q(\cdot|\cdot)$, and auto-regressive draft model $p(\cdot|\cdot)$, initial prompt sequence $x_0, \dots, x_t$.
\STATE Initialise $n \leftarrow t$.
\WHILE{$n < T$}
    \FOR{$t = 1$ \TO $K$}
        \STATE Sample draft auto-regressively $\tilde{x}_t \sim p(x|x_1, \dots, x_n, \tilde{x}_1, \dots, \tilde{x}_{t-1})$
    \ENDFOR
    \STATE In parallel, compute $K + 1$ sets of logits from drafts $\tilde{x}_1, \dots, \tilde{x}_K$:
    \STATE $q(x|x_1, \dots, x_n), q(x|x_1, \dots, x_n, \tilde{x}_1), \dots, q(x|x_1, \dots, x_n, \tilde{x}_1, \dots, \tilde{x}_K)$
    \FOR{$t = 1$ \TO $K$}
        \STATE Sample $r \sim U[0, 1]$ from a uniform distribution.
        \IF{$r < \min\left(1, \frac{q(x|x_1, \dots, x_{n+t-1})}{p(x|x_1, \dots, x_{n+t-1})}\right)$}
            \STATE Set $x_{n+t} \leftarrow \tilde{x}_t$ and $n \leftarrow n + 1$.
        \ELSE
            \STATE sample $x_{n+t} \sim (q(x|x_1, \dots, x_{n+t-1}) - p(x|x_1, \dots, x_{n+t-1}))_+$ and exit for loop.
        \ENDIF
    \ENDFOR
    \IF{all tokens $x_{n+1}, \dots, x_{n+K}$ are accepted}
        \STATE sample extra token $x_{n+K+1} \sim q(x|x_1, \dots, x_n, x_{n+K})$ and set $n \leftarrow n + 1$.
    \ENDIF
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```

This is modified [[Rejection Sampling]]. 

## Why Not Just Sample from $p$ on Rejection?

When a draft token is rejected, you already have the full target distribution $p(\cdot \mid \text{context})$ from the parallel verification pass. Sampling a new token from $p$ is free — no extra forward pass. So why bother with the $(p - q)_+$ correction?

Because sampling from $p$ directly gives the wrong marginal distribution. The marginal probability of outputting token $x$ is the sum of two paths:

$$
 P(\text{output} = x) = \underbrace{\min(p(x), q(x))}_{\text{accepted from draft}} + \underbrace{P(\text{reject}) \cdot p'(x)}_{\text{resampled on rejection}} 
$$

If you set $p'(x) = p(x)$, this becomes $\min(p(x), q(x)) + P(\text{reject}) \cdot p(x)$, which is not equal to $p(x)$ in general. You double-count: tokens where $p$ and $q$ both place mass get delivered through the acceptance path _and_ again through the rejection path, skewing the output.

The $(p - q)_+$ correction fixes this by subtracting out exactly the mass already delivered via acceptance:

$$
 p'(x) = \frac{\max(0,, p(x) - q(x))}{\sum_{x'} \max(0,, p(x') - q(x'))} 
$$

> [!tip] Intuition
> 
> Think of $p$ as a target you need to "fill." The acceptance step already delivers $\min(p(x), q(x))$ of mass for each token. The corrected distribution $(p - q)_+$ contains only the unfilled residual — tokens where $p$ exceeds $q$, weighted by exactly the deficit. The two paths tile $p$ with no overlap.

The normalizing constant $\sum_x \max(0, p(x) - q(x))$ equals $P(\text{reject})$ by conservation of probability ($\sum_x p(x) = \sum_x q(x) = 1$), so the algebra closes:

$$
 \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x) 
$$

This is the unique correction that makes the output distribution exactly $p$ without requiring any additional model calls.