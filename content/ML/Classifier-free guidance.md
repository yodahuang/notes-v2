---
date: 2026-02-05
---

Related: [[Flow Matching]], [[Score matching]].

### Vanilla Guidance

We will let the model take another input, prompt $y$, during training. Instead of sampling the output from $z$, we sample both $z$ and $y$. For example, in training the model, we always provide text "dog" with a dog image, "cat" with a cat image, and so on.

### Classifier Guidance

With some Bayes rule we can separate the $u_t^{\text{target}}(x|y)$ to the unguided part and the guided part. For example, with Gaussian probability paths, we can convert the vector field to its score representation...

$$
u_t^{\text{target}}(x|y) = a_t \nabla \log p_t(x|y) + b_t x
$$

Next, realize that $p_t(x|y)$ is a conditional density. Hence, we can use Bayes' rule to rewrite the guided score as

$$
p_t(x|y) = \frac{p_t(x)p_t(y|x)}{p_t(y)}
$$

$$
\nabla \log p_t(x|y) = \nabla \log \left( \frac{p_t(x)p_t(y|x)}{p_t(y)} \right) = \nabla \log p_t(x) + \nabla \log p_t(y|x)
$$

where we used that the gradient $\nabla$ is taken with respect to the variable $x$, so that $\nabla \log p_t(y) = 0$. We may thus rewrite

$$
u_t^{\text{target}}(x|y) = b_t x + a_t(\nabla \log p_t(x) + \nabla \log p_t(y|x)) = u_t^{\text{target}}(x) + a_t \nabla \log p_t(y|x).
$$

Notice the shape of the above equation: The guided vector field $u_t^{\text{target}}(x|y)$ is a sum of the unguided vector field $u_t^{\text{target}}(x)$ _plus_ a gradient of the likelihood $p_t(y|x)$ of the guidance variable $y$. As people observed that their image $x$ did not fit their prompt $y$ well enough, it was a natural idea to scale up the contribution of the $\nabla \log p_t(y|x)$ term, yielding

$$
\tilde{u}_t(x|y) = u_t^{\text{target}}(x) + w a_t \nabla \log p_t(y|x), \quad (\text{classifier guidance})
$$

Well, where do we get the $y|x$ part? Another classifier, thus the name.

### Classifier-Free Guidance

Well, now you know why we emphasize classifier-free here. Surprise! We used Bayes' rule again, and we got $x|y$ again.

How can we double-dip Bayes rule to get a generative model from classifier, or reuse our current generator to do two jobs? Here it goes

$$
\tilde{u}_t(x|y) = (1-w)u_t^{\text{target}}(x) + wu_t^{\text{target}}(x|y)
$$

Our model can produce both $u(x)$ and $u(x|y)$ since it can treat $u(x) = u(x|\emptyset)$ . We'll hack our label so that with some probability $\eta$ it output this empty thing.