---
aliases: 
Arxiv: https://arxiv.org/abs/2303.09553
pdf: "[[lerf.pdf]]"
original title: "LERF: Language Embedded Radiance Fields"
website: https://www.lerf.io/
---
Distill [[CLIP]] embeddings to 3D space for querying, by training a [[NeRF]] with extra embeddings.
![[lerf_overview.png]]
## Volumetric rendering and training

Recall in [[NeRF]] the input is position $x$ and view direction $d$. For text embedding, in addition to that along the ray, we also take scale $s$ as an input. It's basically a frustum along the ray. The integration works just as before: we not only can integrate color, we can also integrate whatever embedding we want. So in this case, for each sampled point, we just come up with some embedding, volume render it, then compare with the ground truth.

For the ground truth, computing CLIP feature for each point on the ray in different scale on the fly is slow. (Recall in [[NeRF#Hierarchical volume rendering]] the sampled point may change dynamically) . So we just precompute a bunch of them and then do trilinear interpolation.

[[DINO]] is also involved here, without it the relevancy map does not look good. That's understandable as as long as there's something there, CLIP would get some embedding, with no guarantee of edge sharpness for segmentation. So we also try predict DINO features for each pixel and compare with the ground truth one when rendered.

## Querying LERF

When the user provide a query, we return the sampled point and scale that highest relevancy score. 

> To assign each rendered language embedding $\phi_{lang}$ a score, we compute the CLIP embedding of the text query $\phi_{quer}$, along with a set of canonical phrases $\phi^i_{canon}$ . 

[[lerf.pdf#page=6&selection=139,0,158,1|lerf, page 6]]

The score is computed as

$$\min_i\frac{\exp(\phi_\mathrm{lang}\cdot\phi_\mathrm{quer})}{\exp(\phi_\mathrm{lang}\cdot\phi_\mathrm{canon}^i)+\exp(\phi_\mathrm{lang}\cdot\phi_\mathrm{quer}))}$$

> All renderings use the same canonical phrases: “object”, “things”, “stuff”, and “texture”. We chose these as qualitatively “average” words for queries users might mak

[[lerf.pdf#page=6&selection=197,10,211,3|lerf, page 6]]

So this essentially pick "how much closer is rendered embedding towards the query, compared with to the canonical one".

The scale is selected by testing all the score and pick the one with highest score.