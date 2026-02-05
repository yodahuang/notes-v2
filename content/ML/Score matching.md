---
date: 2026-02-04
---

Similar to [[Flow Matching]], this is from that MIT course. 

SDEs are constructed via a **Brownian motion**. 

Let us define it: A **Brownian motion** $W = (W_t)_{0 \leq t \leq 1}$ is a stochastic process such that $W_0 = 0$, the trajectories $t \mapsto W_t$ are continuous, and the following two conditions hold:

1. **Normal increments:** $W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$ for all $0 \leq s < t$, i.e. increments have a Gaussian distribution with variance increasing linearly in time ($I_d$ is the identity matrix).
2. **Independent increments:** For any $0 \leq t_0 < t_1 < \dots < t_n = 1$, the increments $W_{t_1} - W_{t_0}, \dots, W_{t_n} - W_{t_{n-1}}$ are independent random variables.

Brownian motion is also called a **Wiener process**, which is why we denote it with a "$W$".$^1$ We can easily simulate a Brownian motion approximately with step size $h > 0$ by setting $W_0 = 0$ and updating

$$W_{t+h} = W_t + \sqrt{h}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \quad (t = 0, h, 2h, \dots, 1-h)$$

Since it's stochastic and we cannot write derivatives now, we'll just represent it with infinitesimal updates.

$$X_{t+h} = X_t + \underbrace{h u_t(X_t)}_{\text{deterministic}} + \underbrace{\sigma_t (W_{t+h} - W_t)}_{\text{stochastic}} + \underbrace{h R_t(h)}_{\text{error term}}$$

or

$$\begin{aligned} \mathrm{d}X_t &= u_t(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t && \blacktriangleright \text{ SDE} \\ X_0 &= x_0 && \blacktriangleright \text{ initial condition} \end{aligned}$$

**Euler-Maruyama method** is one of the simplest way to simulate it:

$$X_{t+h} = X_t + h u_t(X_t) + \sqrt{h}\sigma_t\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d)$$

We can see $\sigma_{t}=0$ makes it a flow model.

## Score functions

Score functions = gradients of the log-likelihood: $\nabla \log q(x)$ (with respect to x)

For Gaussian path, that's $- \frac{1}{\beta^{2}}x + \frac{\alpha_t}{\beta_t^2}z$ , and it can be reparameterized to vector field.

$$\mathcal{L}(\theta) = \|s_t^\theta(x) - \nabla \log p_t(x|z)\|^2$$