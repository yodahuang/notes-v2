---
aliases: 
Arxiv: 
pdf: "[[transfusion.pdf]]"
original title: "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model"
date: 2025-12-31
tags: 
year: 2024
---
A unified transformer that can output both text and image. Unlike the rather naive ([[Chameleon]]) way, which converts each image to a sequence of discretized tokens and then trains over the combined text-image token sequence, this approach generates image by diffusion, not autoregression. 

> The key difference between Chameleon and Transfusion is that while Chameleon discretizes images and processes them as tokens, Transfusion keeps images in continuous space, removing the quantization information bottleneck. 

[[transfusion.pdf#page=7&selection=118,8,120,37|transfusion, page 7]]

![[transfusion_overview.png]]

## Training

Data: standard LLM corpus + image with captions. It's unclear what's the data schedule.

Loss: $\mathcal{L}_{\mathrm{Transfusion}} = \mathcal{L}_{\mathrm{LM}} + \lambda \cdot \mathcal{L}_{\mathrm{DDPM}}$

The $\mathcal{L}_{\mathrm{LM}}$ is for next token prediction. The $\mathcal{L}_{\mathrm{DDPM}}$ applies to the image part of the sequence. So say the model decides "now is a good time to insert an image here", it would then output a BOI (beginning of image) token, guided by LM loss. It would then keep outputting next token, until it outputs EOI. The generated stuff in between is guided by DDPM loss. 

Let's revisit the above overview image. The input / label pair is not entirely the normal next token prediction pair. Notably, the input image is never "cut in half". Recall that for t size input, GPT-style-LLM outputs t size output, instructed to shift by one. For the image block, the network is instructed to **just** denoise it one step, not next token prediction. See the above image to see how the alignment problem is solved by not outputting EOI token. Note how there's no "next patch prediction".

In inference time, we append $n$ pure noise image patches to the generated sequence once BOI is generated and denoise T steps, manually add EOI (since the model cannot output itself, it never outputs that in training time), and switch to LM mode.

>  Reflecting the training objective, our decoding algorithm also switches between two modes: LM and diffusion. In LM mode, we follow the standard practice of sampling token by token from the predicted distribution. When we sample a BOI token, the decoding algorithm switches to diffusion mode, where we follow the standard procedure of decoding from diffusion models. Specifically, we append a pure noise $x_T$ in the form of n image patches to the input sequence (depending on the desired image size), and denoise over T steps. At each step t, we take the noise prediction and use it to produce $x_{t−1}$, which then overwrites $x_t$ in the sequence; i.e. the model always conditions on the last timestep of the noised image and cannot attend to previous timesteps. Once the diffusion process has ended, we append an EOI token to the predicted image, and switch back to LM mode. This algorithm enables the generation of any mixture of text and image modalities.

[[transfusion.pdf#page=6&selection=83,1,127,88|transfusion, page 6]]


![[transfusion_details.png]]

## Important details

### Image representation

[[U-Net]] on [[Variational Autoencoder (VAE)|VAE]]. It's shown to perform better than just linear.

> Overall, it appears that there are indeed inductive bias benefits to U-Net encoding and decoding of images beyond the mere addition of parameters.

[[transfusion.pdf#page=11&selection=440,86,442,39|transfusion, page 11]]
### Attention mask

See the above image. Of course it's better.
### Image with captions

> Our experiments order 80% of image-caption pairs with the caption first, and the image conditioning on the caption, following the intuition that image generation may be a more data-hungry task than image understanding. The remaining 20% of the pairs condition the caption on the image. However, these images are noised as part of the diffusion objective. We thus measure the effect of limiting the diffusion noise to a maximum of t = 500 (half of the noise schedule) in the 20% of cases where images appear before their captions. Table 8 shows that noise limiting significantly improves image captioning, as measure by CIDEr, while having a relatively small effect (less than 1%) on other benchmarks.

[[transfusion.pdf#page=11&selection=448,0,461,11|transfusion, page 11]]

Less noise for image caption task makes image understanding better. Note that in inference time we always see the real image. In training time the captioning tasks sees noised image since there's also the diffusion loss.
## Not important details
- They used OG DDPM instead of flow matching, and mentions the latter is left for future work
- They use different Context Free Guidance params on different benchmarks (standard practice).

## Results

It performs much better than [[Chameleon]]. 

> One hypothesis is that this stems from the competition between text and image tokens in the output distribution; alternatively, it is possible that diffusion is more efficient at image generation and requires fewer parameters, allowing Transfusion models to use more capacity than Chameleon to model text. We leave further investigation of this phenomenon to future research.

[[transfusion.pdf#page=8&selection=111,45,115,19|transfusion, page 8]]

The image generation is on par with [[SDXL]], lag behind [[SD3]], with data modified to contain more image. They argue that SD3's data is better.