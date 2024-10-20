> Informally, for an input set of real images, we say that the set of layer activations (for any layer $L_i$) forms a “manifold of interest”. It has been long assumed that manifolds of interest in neural networks could be embedded in low-dimensional subspaces. 
[[mobilenet_v2.pdf#page=2&selection=253,13,262,1|mobilenet_v2, page 2]]

Consider the case for ReLU, which in $R^n$ space leads to piece linear curve with n-joints, and thus we got information lost. Here in the image we can see some data is just collapsed in low dim. On higher dim this is fine, since now the manifold is only a small part of the input space.
![[relu_manifold.png]]

Thus, we force the manifold to be low dimension, by having this bottleneck, and we do not have ReLU or other activations on this bottleneck.
![[mobilenetv2_evolution_conv.png]]