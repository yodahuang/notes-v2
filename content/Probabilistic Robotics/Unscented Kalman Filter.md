---
chapter: 3
aliases:
  - UKF
share: true
---

It's like [[Particle Filter|Particle Filter]], but the sample points are determined deterministically. So its' far more efficient when the distribution is Gaussion-like.

While [[Extended Kalman Filter|EKF]] do linearization via first order Tayler expansion, this method does that by something called unscented transform. That means we sample some points around mean, see where they are at in new distribution, and fit a Gaussion on it. Recall in [[Extended Kalman Filter|EKF]] the only thing we care about is the moments. Here it's sampled points, thus more accurate.

The actual thing is as follows:

## Unscented Transform
We choose $2n +1$ sigma points:
![[ukf_formulas.png|ukf_formulas.png]]
$$
w^{[i]}_{m}=w^{[i]}_{c}= \frac{1}{2(n + \lambda)}\quad \text{for} i = 1, ..., 2n. 
$$
The parameter $\beta$ can be chosen to encode additional (higher order) know-  
ledge about the distribution underlying the Gaussian representation. If the  
distribution is an exact Gaussian, then $\beta = 2$ is the optimal choice.
$$
\mathcal{Y}^{[i]} = g(\mathcal{X}^{[i]})
$$
And we can estimate the shape of the resulting Gaussian by:
$$
\begin{aligned}
\mu' &= \sum^{2n}_{i=0} w^{[i]}_{m}\mathcal{Y}^{[i]}\\
\Sigma' &= \sum^{2n}_{i=0} w^{[i]}_{c}(\mathcal{Y}^{[i]} - \mu')(\mathcal{Y}^{[i]} - \mu')^T\\
\end{aligned}
$$
## The UKF
![[ukf.png|ukf.png]]

As you can see here, it's just 3 points sampled.