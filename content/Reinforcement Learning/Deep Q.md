---
aliases:
  - DQN
date: 2025-10-05
updated: 2026-02-16
pdf: "[[285-q-learning-in-practice.pdf]]"
---
Use a NN as the function estimator for Q. The loss is TD loss. 

$$
\delta_t \;=\;
\overbrace{R_{t+1} + \gamma \max_{a} Q(S_{t+1},a)}^{\color{teal}{\text{TD Target}}}
\;-\;
\underbrace{Q(S_t, A_t)}_{\color{blue}{\text{Former Q-value estimation}}}
\quad\; \color{orange}{\text{(TD Error)}}
$$

- A neural network: $O_t \mapsto \mathbf{q}_{\mathbf{w}}$ (action-out)
- An exploration policy: $\pi_t = \epsilon\text{-greedy}(\mathbf{q}_t)$, and then $A_t \sim \pi_t$
- A replay buffer to store and sample past transitions $(S_i, A_i, R_{i+1}, S_{i+1})$
- Target network parameters $\mathbf{w}^{-}$
- A Q-learning weight update on $\mathbf{w}$ (uses replay and target network):

    $$




\Delta\mathbf{w} = \left(R_{i+1} + \gamma \max_{a} q_{\mathbf{w}^{-}}(S_{i+1}, a) - q_{\mathbf{w}}(S_i, A_i)\right) \nabla_{\mathbf{w}}q_{\mathbf{w}}(S_i, A_i)

$$
- Update $\mathbf{w}_t^{-} \leftarrow \mathbf{w}_t$ occasionally (e.g., every 10000 steps)
- An optimizer to minimize the loss (e.g., SGD, RMSprop, or Adam)

But wait, there's still $\max$, so it can't really handle continuous space well. It can handle complicated state space though. 
```pseudo
\begin{algorithm}
\caption{Deep Q-Learning with Experience Replay}
\begin{algorithmic}
  \State Initialize replay memory $D$ to capacity $N$
  \State Initialize action-value function $Q$ with random weights $\theta$
  \State Initialize target action-value function $\hat{Q}$ with weights $\theta^- = \theta$

  \For{episode $= 1, M$}
    \State Initialize sequence $s_1 = \{ x_1 \}$ and preprocessed sequence $\phi_1 = \phi(s_1)$

    \For{$t = 1$ \textbf{to} $T$}
      \Comment{=== Sampling ===}
      \State With probability $\varepsilon$ select a random action $a_t$
      \State otherwise select $a_t = \arg\max_a Q(\phi(s_t), a; \theta)$
      \State Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$
      \State Set $s_{t+1} = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$
      \State Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $D$

      \Comment{=== Training ===}
      \State Sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from $D$
      \If{episode terminates at step $j+1$}
        \State $y_j \gets r_j$
      \Else
        \State $y_j \gets r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-)$
      \EndIf
      \State Perform gradient descent step on $\big(y_j - Q(\phi_j, a_j; \theta)\big)^2$ w.r.t. network parameters $\theta$
      \State Every $C$ steps reset $\hat{Q} = Q$
    \EndFor
  \EndFor
\end{algorithmic}
\end{algorithm}

```

Note how it come from [[Fitted Q Iteration]]. We basically add replay buffer and use transaction from the buffer instead of sampling with the policy.
![[simple_q_learning.png]]

### Several ways to make training stable

Note the experience replay here. Since this is off policy, it can use previous samples to
- Reuse the interaction with the env.
- Avoid forgetting previous experiments and reduce the correlation between experiments

The latter $Q$ part in TD loss can be fixed, so it's a fixed target, more like supervised learning.
 ![[q_learning_with_target_network.png]]

We can also avoid the sudden jump of "copy param every $N$" by using [[Polyak Averaging]] idea: 
$$  

\text{update }\phi':\ \phi' \leftarrow \tau \phi' + (1 - \tau)\phi  

$$
We can linearly interpolate like that in parameter space. 

![[q_learning_general_view.png]]
Additionally, we can have another NN, the Target Network. That's Double DQN.
- Use our **DQN network** to select the best action to take for the next state (the action with the highest Q-value).
- Use our **Target network** to calculate the target Q-value of taking that action at the next state. So it's more like a supervised learning, no moving targets.
See [[Q learning#Double Q-learning]] for more details. 

- standard Q-learning: $y = r + \gamma Q_{\phi'}\!\left(s', \arg\max_{a'} Q_{\phi'}(s', a')\right)$
- double Q-learning: $y = r + \gamma Q_{\phi'}\!\left(s', \arg\max_{a'} Q_{\phi}(s', a')\right)$

Just use current network (not target network) to evaluate action, still use target network to evaluate value!

### N step returns

See [[Temporal difference#Multi-step returns]]. We can use n step return instead of single stage return.
- Less biased target vales when Q-values are inaccurate
- Typical faster training, especially early on.
- Only actually correct when on-policy
- But we can ignore the problem and it seems to still be working well, or dynamically choose N to only on-policy data, or importance sampling
$$

y_{j,t}
=
\sum_{t' = t}^{t + N - 1}
\gamma^{\,t' - t} \, r_{j,t'}
\;+\;
\gamma^{N}
\max_{a_{j,t+N}}
Q_{\phi'}\!\left(s_{j,t+N}, \, a_{j,t+N}\right)

$$

## Continuous Actions

There's this $\max_{a}Q(s, a)$ here that's hard to do for continuous actions
### Stochastic optimization

- We can random sample: sample a bunch in continuous space and pick the max we see.
- Or use cross-entropy method (CEM) or [[CMA-ES]], doing stochastic optimization to "guess the max" basically.

## Use function class that's easy to optimize

That's based on the [[NAF]] paper, with Ilya Sutskever and Sergey Levine in the author list. The basic idea is to use a specific formula that can $max$ easily. 

### Use a NN to tell what's the max

This is the [[DDPG]] idea: train another network $\mu_{\theta}(s)$ such that $\mu_{\theta}\approx\arg \max_{a}Q_{\phi}(s, a)$.
And we do not really train a new network. We just stitch them together, if the network outputs as we hope it outputs, the loss function would just optimize these two together.

You can argue this is quite similar to [[Actor-Critic]]
![[ddpg_actor_critic.png]]

### Practical tips

- Take times, may not stabilize easily
- Large replay buffers help stability
- Start with high exploration.
- Bellman error can be big. We can clip gradients or use Huber loss
- Double Q-learning help *a lot*, no downsides
- N-step return also help a lot, some downsides
- Needs tuning for exploration and learning rates
- Run multiple random seed, very inconsistent
