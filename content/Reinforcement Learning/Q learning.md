Recall dynamic programming algorithms [[Policy & value iteration]] 

$$
\begin{aligned}
v_{k+1}(s) &= \mathbb{E} \left[ R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t = s, A_t \sim \pi(S_t) \right] && \text{(policy evaluation)} \\
v_{k+1}(s) &= \max_a \mathbb{E} \left[ R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t = s, A_t = a \right] && \text{(value iteration)} \\
q_{k+1}(s, a) &= \mathbb{E} \left[ R_{t+1} + \gamma q_k(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a \right] && \text{(policy evaluation)} \\
q_{k+1}(s, a) &= \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} q_k(S_{t+1}, a') \mid S_t = s, A_t = a \right] && \text{(value iteration)}
\end{aligned}
$$

We have the analogous model -free TD algorithms:

$$
\begin{aligned}
v_{t+1}(S_t) &= v_t(S_t) + \alpha_t \left( R_{t+1} + \gamma v_t(S_{t+1}) - v_t(S_t) \right) \quad \text{(TD)} \\
q_{t+1}(s, a) &= q_t(S_t, A_t) + \alpha_t \left( R_{t+1} + \gamma q_t(S_{t+1}, A_{t+1}) - q_t(S_t, A_t) \right) \quad \text{(SARSA)} \\
q_{t+1}(s, a) &= q_t(S_t, A_t) + \alpha_t \left( R_{t+1} + \gamma \max_{a'} q_t(S_{t+1}, a') - q_t(S_t, A_t) \right) \quad \text{(Q-learning)}
\end{aligned}
$$

Of course the value iteration on state $v$ cannot be sampled, so there's no TD algorithm.

Q learning is an [[On and off policy Learning|Off-Policy]] algorithm. It estimates the value of the **greedy** policy  

$$
q_{t+1}(s, a) = q_t(S_t, A_t) + \alpha_t \left( R_{t+1} + \gamma \max_{a'} q_t(S_{t+1}, a') - q_t(S_t, A_t) \right)
$$

**Acting** greedy all the time would not explore sufficiently.

It's soundness depend on a new theorem:

*Q-learning* control converges to the optimal action-value function, $q \rightarrow q^*$, as long as we take each action in each state infinitely often.  

This is different from [[GLIE]]: the policy doesn't need to converge to greedy.
Works for **any** policy that eventually selects all actions sufficiently often (Requires appropriately decaying step sizes $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$,  E.g., $\alpha = 1 / t^{\omega}$, with $\omega \in (0.5, 1)$)  

### A comparison of [[SARSA]] and Q learning

![[cliff_walking_example.png]]
In training time, SARSA gets higher reward since it learns that walking on the edge is dangerous, and it learns a safer path. Recall it's using $\epsilon$-greedy for both prediction and control.
On the other hand, Q learning would stick to the optimal path, since according to its value estimation the optimal path is just the better one, since it has a larger $max$. So it would explore that path more often and got down more often. Note the final policy may be a more optimal one.

### Overestimation
Recall  

$$
\max_a q_t(S_{t+1}, a) = q_t(S_{t+1}, \arg\max_a q_t(S_{t+1}, a))
$$

Uses same values to *select* and to *evaluate*.
... but values are approximate  
- **more** likely to select **overestimated** values  
- **less** likely to select **underestimated** values  

That "max" is persisting. Imagine you are in a state with 100 actions with stochastic outcome. If one of the action by chance got jackpot, then you would keep exploring that state. That leads to super slow convergence: blinded by overestimated values.

#### Double Q-learning
Store two action-value functions: $q$ and $q'$  

$$
R_{t+1} + \gamma q'_t(S_{t+1}, \arg\max_a q_t(S_{t+1}, a)) \tag{1}
$$

$$
R_{t+1} + \gamma q_t(S_{t+1}, \arg\max_a q'_t(S_{t+1}, a)) \tag{2}
$$

Each $t$, pick $q$ or $q'$ (e.g., randomly) and update using (1) for $q$ or (2) for $q'$.  

We can also extend this to SARSA.