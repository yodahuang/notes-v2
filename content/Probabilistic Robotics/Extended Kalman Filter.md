---
chapter: 3
aliases:
  - EKF
share: true
---
A natural extension of [[Kalman Filter|Kalman Filter]].
First, it's non-linear:

![[ekf_comparison.png|ekf_comparison.png]]

Next, we use Jacobian, which is just first order Taylor expansion, so we use $G_t$ and $H_t$ to replace $A_t$, $B_t$ and $C_t$.

The goodness of approximation depends on two things:
- The degree of local non-linearity. Note here it's local. If overall the thing is pretty wild, but around our mean it's linear, that's okay.
- The degree of uncertainty. See this image:
![[uncertainty_affect_goodness.png|uncertainty_affect_goodness.png]]