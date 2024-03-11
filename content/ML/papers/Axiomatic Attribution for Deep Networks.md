---
Created: 2022-02-26T16:07
Link: https://arxiv.org/abs/1703.01365
share: true
---
How do I know how much this input changes with output? Looking at the gradient is good, but it doesn’t handle saddle point (and that’s why we added momentum to SGD). Numerical method has its problem too. This paper uses path integration.

## Two Axioms

- Sensitivity (a): If output is different with only one feature, that feature should be given a non-zero attribution.
- Implementation Invariance: If given the same input, two model always produce the same output, the attributions are always identical.

While the axiom are pretty cool, I doubt that we need that implementation invariance. Is that really important if the input features are correlated with each other?

## Integrated Gradients

$\text{IntegratedGrads}_i(x)::=(x_i - x'_i) \times \int^1_{\alpha = 0}\frac{\partial F(x' + \alpha \times (x - x')))}{\alpha x_i}d\alpha$

It’s just the straight line form of the path methods:

$\text{PathIntegratedGrads}^\gamma_i(x)::=\int^1_{\alpha = 0}\frac{\partial F(\gamma(\alpha))}{\partial\gamma_i(\alpha)}\frac{\partial\gamma_i(\alpha)}{\partial \alpha}d\alpha$

In this case, $\gamma(\alpha) = x' + \alpha \times (x - x')$﻿

It also have property of completeness and others:

$\sum^n_{i=1}\text{IntegratedGrads}_i(x) = F(x) - F(x')$

I’m lazy but it follows some other axioms. Not so important though, as it’s just a fancy line integral to overcome gradient’s problem at saddle points.