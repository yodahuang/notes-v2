---
date: 2025-10-26
---

[[Temporal difference]] learning for action values.

$$q_{t+1}(S_t, A_t) \leftarrow q_t(S_t, A_t) + \alpha \left( \overbrace{\underbrace{\textcolor{blue}{R_{t+1} + \gamma q_t(S_{t+1}, A_{t+1})}}_{\text{target}} - q_t(S_t, A_t)}^{\text{TD error}} \right)$$

It's known as SARSA since it uses $(S_{t}, A_{t}, R_{t+1}, S_{t+1}, A_{t+1})$.