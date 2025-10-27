---
aliases:
  - DQN
date: 2025-10-05
---
Use a NN as the function estimator for Q. The loss is TD loss. 

$$
\delta_t \;=\;
\overbrace{R_{t+1} + \gamma \max_{a} Q(S_{t+1},a)}^{\color{teal}{\text{TD Target}}}
\;-\;
\underbrace{Q(S_t, A_t)}_{\color{blue}{\text{Former Q-value estimation}}}
\quad\; \color{orange}{\text{(TD Error)}}
$$

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

### Several ways to make training stable
Note the experience replay here. Since this is off policy, it can use previous samples to
- Reuse the interaction with the env.
- Avoid forgetting previous experiments and reduce the correlation between experiments

The latter $Q$ part in TD loss can be fixed, so it's a fixed target.

Additionally, we can have another NN, the Target Network. That's Double DQN.
- Use our **DQN network** to select the best action to take for the next state (the action with the highest Q-value).
- Use our **Target network** to calculate the target Q-value of taking that action at the next state. 