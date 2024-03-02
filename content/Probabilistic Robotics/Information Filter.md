---
chapter: 3
share: true
---

The dual of [[Kalman Filter|Kalman Filter]]. 

Instead of using moments to parameterize Gaussian, we use something called canonical parameterization.

$$
\begin {aligned}
\Omega&=\Sigma^{-1}  \\
\xi&=\Sigma^{-1}~\mu
\end {aligned}
$$
The two are called the information matrix and information vector. The benefit is that we can represent log of Gaussian by a quadratic distance weighted by matrix $\Omega$, or, [[Mahalanobis distance|Mahalanobis distance]]. 
$$
	-\log p(x) = \text{const.} + \frac{1}{2}x^{T}\Omega x - x^{T}\xi
$$
![[information_filter.png|information_filter.png]]

For the filter algorithm itself, what's interesting is that compared with [[Kalman Filter|KF]], the prediction step now needs matrix inverse. Previously it's very simple. But in contrast, in the measurement update step it's easier.