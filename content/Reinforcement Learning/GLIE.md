---
date: 2025-11-02
---

### Greedy in the Limit with Infinite Exploration

- All state-action pairs are explored infinitely many times,  

  $$
  \forall s, a \quad \lim_{t \to \infty} N_t(s, a) = \infty


$
$

- The policy converges to a greedy policy,  

  $$

  \lim_{t \to \infty} \pi_t(a \mid s) = \mathcal{I}\left(a = \arg\max_{a'} q_t(s, a')\right)


$$

- For example, $\epsilon$-greedy with $\epsilon_k = \frac{1}{k}$

GLIE Model-free control converges to the optimal action-value function, $q_{t}\rightarrow q_{*}$.