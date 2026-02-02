---
date: 2025-11-02
---

*Goal:* given some function $f$ with random inputs $X$, and a distribution $d'$, estimate the expectation of $f(X)$ under a different (target) distribution $d$.  

*Solution:* weight the data by the ratio $d / d'$  

$$
\begin{aligned}
\mathbb{E}_{x \sim d}[f(x)] 
&= \sum d(x) f(x) \\
&= \sum d'(x) \frac{d(x)}{d'(x)} f(x) \\
&= \mathbb{E}_{x \sim d'} \left[ \frac{d(x)}{d'(x)} f(x) \right]
\end{aligned}
$$

This is useful in [[On and off policy Learning|Off-Policy]] learning.
Ergo, when following policy $\mu$, can use $\frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)} R_{t+1}$ as unbiased sample.  
## For Off-Policy Monto Carlo:
*Goal:* estimate $v_\pi$  
*Data:* trajectory $\tau_t = \{ S_t, A_t, R_{t+1}, S_{t+1}, \ldots \}$ generated with $\mu$  
*Solution:* use return $G(\tau_t) = G_t = R_{t+1} + \gamma R_{t+2} + \ldots$, and correct:  

$$
\begin{aligned}
\frac{p(\tau_t | \pi)}{p(\tau_t | \mu)} G(\tau_t)
&= \frac{p(A_t | S_t, \pi)p(R_{t+1}, S_{t+1} | S_t, A_t)p(A_{t+1} | S_{t+1}, \pi) \cdots}
{p(A_t | S_t, \mu)p(R_{t+1}, S_{t+1} | S_t, A_t)p(A_{t+1} | S_{t+1}, \mu) \cdots} G_t \\[6pt]
&= \frac{\pi(A_t | S_t) \pi(A_{t+1} | S_{t+1}) \cdots}
{\mu(A_t | S_t) \mu(A_{t+1} | S_{t+1}) \cdots} G_t
\end{aligned}
$$

### For TD

$$
v(S_t) \leftarrow v(S_t) + \alpha \left( 
\frac{\pi(A_t \mid S_t)}{\mu(A_t \mid S_t)} 
\big( R_{t+1} + \gamma v(S_{t+1}) \big) - v(S_t)
\right)
$$

### For SARSA
This is called **Expected SARSA**
- No importance sampling is required  
- Next action may be chosen using behaviour policy $A_{t+1} \sim \mu(\cdot \mid S_{t+1})$  
- But we consider probabilities under $\pi(\cdot \mid S_t)$  
- Update $q(S_t, A_t)$ towards value of alternative action  

$$
q(S_t, A_t) \leftarrow q(S_t, A_t)
+ \alpha \left(
{
\color{#1E90FF}R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) q(S_{t+1}, a)}
- q(S_t, A_t)
\right)
$$

So you can see the expectation there instead of the $\max$ in Q. Actually, Q-learning is a special case with greedy target policy $\pi$.
