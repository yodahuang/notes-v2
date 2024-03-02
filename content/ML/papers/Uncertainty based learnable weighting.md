---
share: true
---
- [Arxiv link](https://arxiv.org/abs/1705.07115)
- [[multi_task_uncertainty_weighting.pdf |PDF link]]

This provide a way for self-paced multitask learning. Or, for a model that have muitiple task, how to weight the loss terms. If picked right, a model with multi-task target can perform better than focusing on a single task.

This method uses "task uncertainty", which is 

> Task-dependent or Homoscedastic uncertainty is aleatoric uncertainty which is not dependent on the in- put data. It is not a model output, rather it is a quantity which stays constant for all input data and varies between different tasks. It can therefore be described as task-dependent uncertainty.

So basically even if the input is the same, the output would vary, and that's this uncertainty.

## The formula
Very very simple. Recall regression loss, MSE (mean squared error) can be intepreted as MLE of Gaussian. Let $f^W(x)$ be the output of a neural network with weights $W$ on input $x$, then
$$
log\ p(y|f^W(x)) \propto - \frac{1}{2\sigma^2} ||y - f^W(x)||^2 - log\ \sigma
$$
So if we have two regression target, it's 
$$
\frac{1}{2\sigma_1^2}\mathcal{L}_1(W) + \frac{1}{2\sigma_2^2}\mathcal{L}_2(W) + log\sigma_1\sigma_2
$$
We learn the $\sigma_1$ and $\sigma_2$ together with our usual $W$. Note that they cannot be too big because the last term is regulating them.

If there's classification, we can represent it with temperatured softmax, which is a Boltzman distribution:
$$
p(y|f^W(x), \sigma) = \text{Softmax}(\frac{1}{\sigma^2}f^W(x))
$$
And you'll get another formula. See the [[pdf/multi_task_uncertainty_weighting.pdf#page=5|OG paper]] for details.