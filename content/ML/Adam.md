---
date: 2026-03-15
aliases:
  - AdamW
---
Based on my conversation with Claude Sonnet 4.6

---
## Adam

Adam (Adaptive Moment Estimation) combines two ideas: **momentum** and **adaptive per-parameter learning rates**.

### First Moment — Momentum

Instead of following the raw gradient, maintain a running average of past gradients:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

This smooths out noisy gradient directions and accelerates movement along consistent directions.

### Second Moment — Adaptive Learning Rates

Maintain a running average of squared gradients:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$v_t$ estimates $\mathbb{E}[g^2]$ — the typical squared gradient magnitude for each parameter. Its square root $\sqrt{v_t}$ is a proxy for the **typical gradient scale** of that parameter.

### The Update Rule

$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot m_t$$

This can be rewritten as:

$$\Delta\theta \propto \frac{m_t}{\sqrt{v_t}} \approx \frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}}$$

> [!question] Why divide by $\sqrt{\mathbb{E}[g^2]}$ and not $\mathbb{E}[g]$?
> 
> Dividing by $\mathbb{E}[g]$ would just compute something close to a sign — you'd lose all magnitude information and every parameter would take the same effective step size.
> 
> Dividing by $\sqrt{\mathbb{E}[g^2]}$ instead gives a **unit-variance normalized** gradient. The ratio $\frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}}$ is bounded in $[-1, 1]$ by Cauchy-Schwarz, so steps are controlled, but the scale of $\sqrt{\mathbb{E}[g^2]}$ still carries real information about how active that parameter's gradients are.
> 
> It's also reminiscent of a **signal-to-noise ratio**: numerator is the mean signal, denominator is the RMS amplitude.

### Intuition: Crude Diagonal Curvature Estimate

In second-order optimization (Newton's method), you'd divide by the Hessian:

$$\Delta\theta = -H^{-1} g$$

The Hessian is expensive. But $\mathbb{E}[g^2]$ is a cheap, diagonal, gradient-based proxy:

- Large, varying gradient → high curvature → smaller steps
- Tiny gradient → flat landscape → bigger steps

Adam is essentially doing **diagonal quasi-Newton** without computing second derivatives.

### Why This Matters for Sparse Parameters

Consider an embedding table. Most tokens appear rarely, so most rows get zero gradient most steps. With SGD, those rows barely move. With Adam, their $v_t$ stays near zero, so $1/\sqrt{v_t}$ is large — rare but informative updates land with meaningful step size.

---

## Bias Correction

### The Problem

Both moments are initialized to zero: $m_0 = 0,\ v_0 = 0$.

After the first step:

$$m_1 = \beta_1 \cdot 0 + (1-\beta_1) g_1 = (1-\beta_1) g_1$$

With $\beta_1 = 0.9$, that's $m_1 = 0.1 \cdot g_1$ — massively underestimating the true gradient. The zero initialization **bleeds into the estimate**.

### The Fix

After $t$ steps, the running average satisfies:

$$\mathbb{E}[m_t] = \mathbb{E}[g] \cdot (1 - \beta_1^t)$$

Divide it out:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

Now $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]$. Unbiased. Same correction applies to $v_t$ with $\beta_2$.

### It Vanishes Quickly

With $\beta_1 = 0.9$:

|$t$|$1 - \beta_1^t$|
|---|---|
|1|0.10|
|5|0.41|
|10|0.65|
|50|0.995|

By step 50 the correction is essentially 1. It only matters in the first few dozen steps — but those early steps can be consequential, especially with warmup schedules.

> [!note] $\beta_2$ bias lingers longer
> 
> Since $\beta_2 = 0.999$ is closer to 1, the bias in $v_t$ decays more slowly than in $m_t$. Still vanishes, just takes a bit longer.

---

## AdamW

### The Problem with Adam + Weight Decay

Naively, L2 regularization adds $\lambda\theta$ to the gradient, which then gets divided by $\sqrt{v_t}$. For parameters with large gradients, this **weakens the regularization** — the penalty gets adaptively scaled down along with everything else.

### The Fix: Decouple Weight Decay

$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t - \eta \lambda \theta$$

The weight decay term $\eta\lambda\theta$ is applied **directly to the weights**, not mixed into the gradient before adaptive scaling. Every parameter gets shrunk by the same proportional amount regardless of gradient history.

> [!summary] Intuition
> 
> In Adam, weight decay gets "laundered" through the adaptive scaling and loses its regularization effect. AdamW ensures the penalty is always proportional to the weight magnitude — which is what regularization is supposed to do.

### Why AdamW Dominates in Practice

For large models, weight decay is a primary regularization tool. AdamW's decoupling makes it actually behave like L2 regularization is supposed to. Essentially all modern LLM training uses AdamW (or variants like Lion, Adafactor).

**Typical defaults:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, with $\lambda$ tuned per model.

## My own implementation

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], eps: float, weight_decay: float):
        assert lr > 0
        assert betas[0] > 0 and betas[1] > 0
        assert 0 < eps < 1.0
        assert weight_decay > 0
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Use lazy initialization pattern from official PyTorch
                if len(state) == 0:
                    state["running_first_moment"] = torch.zeros_like(p)
                    state["running_second_moment"] = torch.zeros_like(p)
                    state["t"] = 1

                # self.state is a default dict with Tensor key
                state = self.state[p]
                # if seems that state is a dict by default
                grad = p.grad.data
                state["running_first_moment"] = betas[0] * state["running_first_moment"] + (1 - betas[0]) * grad
                state["running_second_moment"] = betas[1] * state["running_second_moment"] + (1 - betas[1]) * (
                    grad * grad
                )
                # Bias correction for early steps
                # Offsets the zero initialization
                t = state["t"]
                adjusted_lr = lr * math.sqrt(1 - (betas[1] ** t)) / (1 - betas[0] ** t)
                p.data -= (
                    adjusted_lr * state["running_first_moment"] / (torch.sqrt(state["running_second_moment"]) + eps)
                )
                # Weight decay decoupled
                p.data -= lr * weight_decay * p.data
                state["t"] += 1

        return loss

```