---
chapter: 2
share: true
aliases: [Bayes Filter]
---

- $z_t$: measurement data
- $u_t$: control data
- $x_t$: state

## Probabilistic Generative Laws (how we simplify the things) 
### State transition probability

$$
p(x_t|x_{0:t-1}, z_{1:t-1}, u_{1:t}) = p(x_t|x_{t-1}, u_t)
$$
This means only $u_t$ matters if we know $x_t-1$. 
### Measurement probability
$$
p(z_t|x_{0:t}, z_{1:t-1}, u_{1:t}) = p(z_t|x_t)
$$
This means $x_t$ is sufficient to predict measurement $z_t$.

More notations:
$$bel(x_{t}) = p(x_t|z_{1:t}, u_{1:t})$$
If we try to **predict** $x_t$ before we got measurement $z_t$, we got the posterior
$$
\overline{bel}(x_{t})=p(x_{t}|z_{1:t-1},u_{1:t})
$$
From $\overline{bel}(x_t)$ to $bel(x_t)$ is called *correction* or *measurement update*.

## Bayes Filter Algorithm
![[bayes_filter.png]]

So unlike our previous definition, we now compute $bel(x_t)$ in a recursive fashion. 
In line 3, we do predict: given control, sum over all the probability assuming different $x_{t-1}$
In line 3, we do measurement update, take in $z_t$ and get the conditional probability on that. Obviously some Bayes rule kicks in.