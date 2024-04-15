---
share: true
---
mixup: Beyond Empirical Risk Minimization
- [Arxiv link](https://arxiv.org/abs/1710.09412)
- [Another study on calibration](https://arxiv.org/pdf/1905.11001.pdf)

Add in virtual training examples
$$
\begin{aligned}
\tilde{x} &= \lambda x_i + (1 - \lambda) x_j \\ 
\tilde{y} &= \lambda y_i + (1 - \lambda) y_j
\end{aligned}
$$
$x_i$, $x_j$ are raw input vectors, $y_i$, $y_j$ are one-hot label encodings. $\lambda\sim\text{Beta}(\alpha, \alpha)$. $\alpha\in[0.1, 0.4]$ gave the best result for classification.

The important stuff is here label need to be mixed too, so not simple data augmentation. It makes everything better, from classification result to confidence score.