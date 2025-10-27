---
aliases:
  - SE
Arxiv: https://arxiv.org/abs/1709.01507
pdf: "[[squeeze_and_excitation.pdf]]"
original title: Squeeze-and-Excitation Networks
date: 2025-02-07
tags: 
year: 2019
---
Simple idea stretched long.

CNN, by its design, can only focus on local features. This new block enables a style of self attention that is fast and easy to replace blocks in CNN. 

So first you do *squeeze*, which is use a global average pooling per channel, across $H \times W$, call it $z_c$.
Then you do excitation, which is adaptive recalibration:

$$
s = F_{ex}(z, W) = \sigma(g(z, W)) = \sigma(W_2\delta(W_1z))
$$

Here $\delta$ is ReLU, $\sigma$ is Sigmoid. $W_{1}\in\mathbb{R}^{\frac{C}{r}\times C}$, $W_{2}\in\mathbb{R}^{C\times\frac{C}{r}}$ . This is just a dimension reduction trick so it runs faster. $r=16$ leads to good enough result. One can see this is basically the simpler idea of a transformer attention block, where there's also dimension reduction and sigmoid (but scaled).
After that, we just multiply the "weight" to the original feature map.

### Ablation
- The choice of the squeeze operator doesn't matter much (though the tested ones are all pretty simple)
- The SE block can be put before or after normal block, or inside.
- In the early layers SE blocks show similar excitation across class, and it differs in deeper layers.
- Later layers may be less important (but also more computationally expensive).

It's also mentioned that
>  We found empirically that on ResNet architectures, removing the biases of the FC layers in the excitation operation facilitates the modelling of channel dependencie

[[squeeze_and_excitation.pdf#page=8&selection=423,22,426,11|squeeze_and_excitation, page 8]]