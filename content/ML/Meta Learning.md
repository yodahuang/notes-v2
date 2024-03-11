---
share: true
---
This summary is based off [this blog](https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0)

This mainly pocus on [[MAML]] and [[Reptile]]

> The goal of both of these papers is to solve the _K-shot_ learning problem. In K-shot learning, we need to train a neural network to generalize based on a very small number of examples (often on the order of 10 or so) instead of the often thousands of examples we see in datasets like ImageNet.
> 
> However, in preparation for K-shot learning, you are allowed to train on many similar K-shot problems to learn the best way to generalize based on only K examples.
> 
> The metalearning approach of both Reptile and MAML is to come up with an **initialization** for neural networks that is easily generalizable to similar tasks.

Why not transfer learning? Freeze some layer and just train. Well, it may not learn optimally. For example, if we want to learn a function $f(x) = a\sin(x + b)$, over a range of $a$ and $b. If we sample some and train a model, the model may just learn to output $y=0$ cause that's the average.