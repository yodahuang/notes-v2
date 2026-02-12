---
date: 2026-02-11
pdf: "[[285-value.pdf]]"
---
[[Policy & value iteration|Value Iteration]], but with a NN to fit it.
![[fitted_value_iteration.png]]
The difference between this and the vanilla one is with new evidence, we add that to data instead of doing simple table value setting.

Note that for policy & value iteration, we assume that we can freely explore the environment. If we don't know the transition dynamics, that $\max_{a_i}$ is hard.

Thus we have [[Fitted Q Iteration]]