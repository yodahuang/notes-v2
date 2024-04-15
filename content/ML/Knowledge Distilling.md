---
Created: 2021-06-05T16:21
Link: https://arxiv.org/abs/1503.02531
share: true
---
A simple paper (it’s written in 2015!) The authors does not have the notion of teacher and student network but it’s just like that. We can use the output of teacher network as “soft target” of the student network.

For tasks like MNIST in which the cumbersome model almost always produces the correct answer with very high conﬁdence, much of the information about the learned function resides in the ratios of very small probabilities in the soft targets.

Our more general solution, called “distillation”, is to raise the temperature of the ﬁnal softmax until the cumbersome model produces a suitably soft set of targets. We then use the same high temperature when training the small model to match these soft targets.