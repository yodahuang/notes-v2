---
share: true
---
# ConvNet for the 2020s
- [Arxiv link](https://arxiv.org/abs/2201.03545)
- [[pdf/convnext.pdf|pdf link]]

An aggregation of tricks / designs on ResNet so it can outperform [[Swin Transformer|Swin Transformer]]. The authors gradually add in more and more "modernization" pieces.
![[convnext_overview.png|convnext_overview.png]]

## Training techniques
- Close to [[DeiT|DeiT]] and [[Swin Transformer|Swin Transformer]].
- 300 epochs instead of 90
- [[AdamW|AdamW]] for optimizer
- Augmentation: [[Mixup|Mixup]], [[cutmix|cutmix]], [[RandAugment|RandAugment]], [[RandomErasing|RandomErasing]]
- Regularization: [[Stochastic Depth|Stochastic Depth]], [[Label Smoothing|Label Smoothing]].

## Macro design
- Stage ratio: Adjust the number of blocks in each stage from $(3, 4, 6, 3)$ to $(3, 3, 9, 3)$. This is adapted from Swin-T's similar numbers. It uses $1:1:3:1$, and for larger ones, $1:1:9:1$.
- "patchify stem": [[Vision Transformer (ViT)|ViT]]'s style is like kernel size (14 or 16 for a patch) with non-overlapping convolutinon. [[Swin Transformer|Swin Transformer]]'s approach is smaller ones for multi-stage. Here we use a similar approach, $4\times 4$ stride $4$ convolution layer (non-overlapping), for stem which is at the network's beginning.
Comment: We'll see [[#Large kernel size| > Large kernel size]] later. The $4 \times 4$ mentioned here is the how many patches [[Swin Transformer|Swin Transformer]] splits in an image, but each patch is $7 \times 7$. Plus, the performance doesn't change much. I don't think this is something good.

## ResNext-ify
- Use [[ResNext|ResNext]]'s idea of grouped convolution. Here we use depthwise convolution, a special case for grouped convolution where the number of groups equals the number of channels. The result is increased FLOPs.

## Inverted bottleneck
- Transformers have it. So we're gonna have this too.(Though it doesn't bring much performance gain, and make the network runs slower).
- In the following (c), it's the configuration after we do something afterwards. More random.
![[convnext_bottleneck.png|convnext_bottleneck.png]]
Comment: Another nonsense, only make things slower. 

## Large kernel size
- So larger kernel size means more computation, so we adapt that (c) above. The authors argue this is also learned from Transformers: self attention layer is prior to MLP.
- And they tried different kernel size, and... $7 \times 7$ is the best.
Comment: what a *surprise*. I mean [[Swin Transformer|Swin Transformer]] is $7 \times 7$. 

## Micro design
- Replace RELU with GELU. Exactly the same accuracy.
- Delete GELU in the blocks except for one. Interestingly improves performance. This is suggested by Transformers has fewer activation functions.
- Also delete a BatchNorm layer. Slight improvement.
- Replace BachNorm with [[LayerNorm|LayerNorm]]. With all the changes previously we got a better performance now.
- Separate out the downsampling layers. Between each stage, $2\times2$ conv layers with stride $2$ for spatial downsampling. For [[ResNet|ResNet]] it was $3\times 3$, at the start of each stage. To stabalize training, some [[LayerNorm|LayerNorm]] layers are added. 
![[convnext_block.png|convnext_block.png]]

![[convnext_arch_comparison.png|convnext_arch_comparison.png]]