---
date: 2026-03-15
pdf: "[[cs336_lecture_03.pdf]]"
---
Based my chat with ChatGPT (version unknown) and organized by Claude Opus 4.6.

---

## Formulas

**Original GLU** (Dauphin et al., 2017):

$$
\text{GLU}(x) = \sigma(W_1 x) \odot W_2 x
$$

**SwiGLU** (Shazeer, 2020; used in LLaMA, PaLM, etc.):

$$
\text{SwiGLU}(x) = W_2 \big(\text{SiLU}(W_1 x) \odot W_3 x\big)
$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$.

BTW that Shazeer paper just have one author and three pages.

> [!question] What's the gate in SwiGLU? 
> In GLU, $W_1$ is unambiguously the gate — sigmoid produces a $[0,1]$ mask. In SwiGLU, the $\text{SiLU}(W_1 x)$ path can look like a traditional activation (à la ReLU), which makes $W_3 x$ seem like it might be the gate. But the roles didn't flip: **$W_1$ still plays the gating/modulation role** via SiLU (a smooth, non-saturating gate), $W_3 x$ is the modulated feature, and $W_2$ is just the standard FFN down-projection.

## Key conceptual shift: multiplicative entanglement

In a standard ReLU/GELU FFN, linear and nonlinear parts are cleanly separable:

$$
f(x) = W_{\text{down}} \cdot \phi(W_{\text{up}} , x)
$$

Each hidden neuron is activated independently — the nonlinearity just thresholds or warps each dimension on its own.

In SwiGLU, the nonlinearity is **multiplicative between two learned projections**:

$$
f(x) = W_2 \big(\phi(W_1 x) \odot W_3 x\big)
$$Each hidden unit becomes $\phi(a_i(x)) \cdot b_i(x)$ — a product of two different linear functions of the input. This creates second-order feature interactions _before_ the down-projection, which is fundamentally more expressive than elementwise activation alone.

|Aspect|ReLU/GELU FFN|SwiGLU FFN|
|---|---|---|
|Structure|Linear → Activation → Linear|Two parallel linears → Multiply → Linear|
|Nonlinearity role|Independent per-neuron thresholding|Multiplicative modulation between features|
|Interaction type|Additive|Bilinear / second-order|
|Separation|Clean (linear ↔ nonlinear separable)|Entangled (activation modulates a learned feature)|

**Intuition:**

- ReLU/GELU: "Turn neurons on or off, then mix."
- SwiGLU: "Smoothly rescale one learned feature using another, then mix."

This multiplicative pattern echoes gated RNNs (LSTM/GRU), attention (Q·K modulates V), and FiLM conditioning — modern deep learning keeps converging on the idea that **multiplicative modulation > pure additive nonlinearity**.

> [!note] Parameter cost
> SwiGLU has 3 weight matrices instead of 2, so at equal `d_ff` it's 50% more parameters. In practice LLaMA compensates by using $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$ (rounded to a multiple of 256) instead of the standard $4 \times d_{\text{model}}$.

## PyTorch implementation

```python
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        w1_x = self.w1(x)
        soft_gate = w1_x * torch.sigmoid(w1_x)  # SiLU(W1 x)
        up_projection_value = soft_gate * self.w3(x)  # SiLU(W1 x) ⊙ W3 x
        return self.w2(up_projection_value)  # Down projecting
```

Note that `w1_x * torch.sigmoid(w1_x)` is manually computing $\text{SiLU}$. You could equivalently use `F.silu(self.w1(x))`.