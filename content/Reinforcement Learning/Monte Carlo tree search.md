---
aliases:
  - MCTS
date: 2026-02-20
---

We are basically building a tree given discrete actions. We switch to other policies if the tree grows too large.

![[mcts.png]]
![[mcts_detailed.png]]
You can see that the policy here is basically [[Multi-arm bandits#UCB|UCB]]. Recall the formula there is

$$
a_t=\underset{a\in\mathcal{A}}{\operatorname*{\operatorname*{argmax}}}Q_t(a)+c\sqrt{\frac{\log t}{N_t(a)}}
$$

The difference is $t$ vs $N(s_{t-1})$. UCT can be understood simply as applying UCB on each tree node (use parent instead of global count). In bandit case there's just one state, but now we have multiple.