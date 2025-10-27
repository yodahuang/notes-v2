---
date: 2025-10-18
---


I'm so disappointed In the HuggingFace online course, I would start my notes from scratch with [DeepMind x UCL 2021 lecture series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm).

## The why

The reward hypothesis:
> Any goal can be formalized as the outcome of maximizing a communal award reward. 

I would also like to put this image here from [[LLM Post-Training]] that I find helpful:
![[rl_domain.png]]

## State
As we can see from control theory, the definition of state can be crucial. There are environment state and agent state. Let's talk about agent state here.
The *history* is
$$H_{t}= O_{0}, A_{0}, R_{1}, O_{1}, \dots, O_{t-1}, A_{t-1}, R_{t,}O_t$$
The markovian agent state $S_t$ typically is some *compression* of $H_t$. The full hitory is Markov (but keeps growing).
In an environment with partial observability, the observations are not Markovian, making them a POMDP. The environment state may still be Markovian, but the agent does not know it. 

## Value function
A quick reminder: we need the notion of value / state, since the reward only have meaning in a state. It's in a specific state that we receives reward, not with a specific action. In other words, state is how we *define* the state where we could get reward.

The *actual* value function is defined as

$$\begin{aligned}v_{\pi}(s)&=\mathbb{E}\left[G_{t}\mid S_{t}=s,\pi\right]\\&=\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+...\mid S_{t}=s,\pi\right]\end{aligned}$$

$\gamma$  $0$ means we only care about the instant reward, while $1$ means it considers long term and immediate rewards equally. Personally I'm interested to see if there are alternative definitions. There's no reason it's defined like this.

The bellman equation is its recursive form, the [[Bellman equation]].

$$
\begin{align*}
v_{\pi}(s) &= \mathbb{E} \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t \sim \pi(s) \right] \\
&= \mathbb{E} \left[ R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t \sim \pi(s) \right]
\end{align*}
$$

Now note here that $a$ is chosen by policy $\pi$ in state $s$.
We can take out the dependency on a specific policy by stating that the following holds for the optimal case: 

$$v_*(s)=\max_a\mathbb{E}\left[R_{t+1}+\gamma v_*(S_{t+1})\mid S_t=s,A_t=a\right]$$

## model
A *model* predicts what the environment will do next. It can predict the next state, or the next (immediate reward). Note it does not directly give a policy.

$$
\begin{aligned}
P(s,a,s^{\prime})&\approx p(S_{t+1}=s^{\prime}\mid S_t=s,A_{t=a)} \\
\mathcal{R}(s,a)&\approx\mathbb{E}\left[R_{t+1}\mid S_t=s,A_t=a\right]
\end{aligned}
$$

## Prediction and control
This is like the Expecation-maximization algorithm. 
- *Prediction*: evaluate the future (for a given policy)
- *Control*: optimise the future (find the best policy)