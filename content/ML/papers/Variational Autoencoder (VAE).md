---
aliases: [VAE]
year: 2013
share: true
---
- [Arxiv Link](https://arxiv.org/abs/1312.6114)
- [Blog post](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

Note the OG paper was not read though. I just read the blog post.

Auto encoder: easy, but may not produce what we want

  

![https://miro.medium.com/max/3608/1*iSfaVxcGi_ELkKgAG0YRlQ@2x.png](https://miro.medium.com/max/3608/1*iSfaVxcGi_ELkKgAG0YRlQ@2x.png)

  

A variational autoencoder can be defined as being an autoencoder whose training is regularised to avoid overfitting and ensure that the latent space has good properties that enable generative process.

  

![https://miro.medium.com/max/3608/1*ejNnusxYrn1NRDZf4Kg2lw@2x.png](https://miro.medium.com/max/3608/1*ejNnusxYrn1NRDZf4Kg2lw@2x.png)

  

![https://miro.medium.com/max/3096/1*Q5dogodt3wzKKktE0v3dMQ@2x.png](https://miro.medium.com/max/3096/1*Q5dogodt3wzKKktE0v3dMQ@2x.png)

  

We want the distribution of encoder similar to a standard normal distribution (input → distribution → almost normal).

  

![https://miro.medium.com/max/3742/1*9ouOKh2w-b3NNOVx4Mw9bg@2x.png](https://miro.medium.com/max/3742/1*9ouOKh2w-b3NNOVx4Mw9bg@2x.png)

  

Note that $\mu$ and $\sigma$ are both multi-dimension embeddings.

  

![https://miro.medium.com/max/3328/1*eRcdr8gczweQHk--1pZF9A@2x.png](https://miro.medium.com/max/3328/1*eRcdr8gczweQHk--1pZF9A@2x.png)
