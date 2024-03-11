---
aliases: [OG Transformer]
share: true
---
- [Arxiv link](https://arxiv.org/abs/1706.03762)

- The use of self attention inside both encoder and decoder itself, not only encoder-decoder level “normal” attention
- The clever positional encoding with sin and cos waves, and use residual blocks to propagate that information. You can think of it as a more concise way of binary encoding (for floats).

[Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

- Multi-head attention is kinda like general attention where a linear layer is used when combining key and query, but probably better, as we could have multiple key, value, query now.

![[transformer.png]]
![[transformer_1.png]]

[Master Positional Encoding: Part I](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3)