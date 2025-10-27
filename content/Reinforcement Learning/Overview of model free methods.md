---
date: 2025-10-26
---

Model free means we are going to do sampling because we cannot enumerate what my next state would be. We only know it when we run it. 
Recall for known model, we have [[Policy & value iteration]], for model free, we have the "sampling version" for them. 

It should also be noticed that we will need function approximation because we need generalization. So we will first have a loss function or objective function, and then we'll apply stochastic gradient descent to update our parameters. 

The very basic bias-free method is the [[Monte Carlo]]. 
An overview of the methods can be found in [[Temporal difference]].