---
share: true
---
# MLP-Mixer: An all-MLP Architecture for Vision
- [Arxiv link](https://arxiv.org/abs/2105.01601)
- [[pdf/mixer.pdf|pdf link]]

Make MLP-only network competitive with CNN and Transformers, even on large scale data.
- It's not there with SOA CNN-with-attention yet, but close.
- Inference 3 times faster than [[Vision Transformer (ViT)|ViT]], similar to CNNs.
- So it seem to prove designed right, MLP could be great, but just on par with best CNN.
## Design

![[mixer_overview.png|mixer_overview.png]]

1. Divide the image into $S$ non-overlapping patches, project to hidden dimension $C$. That's the colorful bits in the figure.
2. Apply Mixer layers, which contains two MLP blocks:
	1. Token mixing MLP: Mix across different spatial locations (tokens). It mixes channel 1 of all the locations together, 2 together ... It can be viewed as a single channel depth-wise convolutions of a full receptive field and parameter sharing. The parameter of this is shared across channels. This is not common, but the result is good so 🤷.
	2. Channel mixing MLP: Mix within the token. It can be viewed as a $1\times1$  convolutions. Again, parameter sharing.
## Experiments
- This is easier to overfit so we need extra regularization. But its performance improve fast with the increase of training data.