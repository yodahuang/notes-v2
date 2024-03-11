---
Created: 2021-10-26T10:42
Link: https://arxiv.org/abs/1711.07971
share: true
---
An easy paper. The idea mainly come from the Non-local means method.

> [!info] Non-local means - Wikipedia  
> Non-local means is an algorithm in image processing for image denoising.  
> [https://en.wikipedia.org/wiki/Non-local_means](https://en.wikipedia.org/wiki/Non-local_means)  

## The OG Non-local means

I'll recap the OG idea here (which is surprisingly new, from 2005). Instead of denoising image with simple Gaussian blur (which is basically a exponentially weighted local mean), sample points from all the places and do the mean. The weight shall represent "how closely related are these two pixels". It's often implemented just as the distance of "local mean value surrounding that pixel".

So now here comes our paper, which just states "let's use this method, but not only for denoising purpose, and let's let the weight learnable".

## The block

A non-local block:

$y_i = \frac{1}{C(x)}\sum_{\forall j}f(x_i, x_j)g(x_j) \\$

Here $i$﻿ is the position we are looking at. When implementing, $g(x_j) = W_gx_j$﻿, and that $f$﻿ can be Gaussian, Embedded Gaussian, Dot product, etc. Refer to the paper for more details. Importantly, self-attention is just a special case of the embedded gaussian non-local operation.

## Difference between this and FC

A fully connected layer is basically doing pattern matching, solely based on location. It's not translation / rotation invariant. This one is, as it's a global function based on the value between pixel values.

> The non-local operation is also different from a fully connected (fc) layer. Eq.(1) computes responses based on relationships between different locations, whereas fc uses learned weights. In other words, the relationship between $x_j$﻿ and $x_i$﻿ is not a function of the input data in fc, unlike in nonlocal layers. Furthermore, our formulation in Eq.(1) supports inputs of variable sizes, and maintains the corresponding size in the output. On the contrary, an fc layer requires a fixed-size input/output and loses positional correspondence (e.g., that from $x_i$﻿ to $y_i$﻿ at the position $i$﻿).

## Questions

Some unanswered question in this paper:

- The network always start from a pretrained model, and then insert the non-local block in, with $W_z$﻿ initialized to $0$﻿, so that the network behaves exactly the same as before. What's unclear is what would happen if we train from scratch.