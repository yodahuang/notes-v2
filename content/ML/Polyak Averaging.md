
This is generated with my conversation with ChatGPT 5.2.

---
## Core Idea

> [!definition]
> Polyak Averaging (a.k.a. Exponential Moving Average, EMA) updates parameters via:
>
> $$
> \theta' \leftarrow \tau \theta' + (1-\tau)\theta
> $$
>
> This is a **linear interpolation in parameter space**, not in function space.

We are averaging weight vectors in $\mathbb{R}^d$.  
The resulting function is **not** a linear combination of network outputs — only the parameters are averaged.

---

## What EMA Really Does

Unrolling the recursion:

$$
\theta'_t
=
(1-\tau)\theta_t
+
\tau(1-\tau)\theta_{t-1}
+
\tau^2(1-\tau)\theta_{t-2}
+
\cdots
$$

> [!intuition]
> EMA is a **low-pass filter over the training trajectory**.
> It smooths SGD noise and tracks a moving average of recent models.

---

## Geometric Interpretation

Modern neural networks are highly overparameterized.

Empirically:

- Many minima are connected by low-loss paths.
- Solutions often lie inside **wide flat valleys**, not sharp isolated pits.

> [!insight]
> SGD tends to wander inside a flat basin.
> EMA pulls the solution toward the **center of that basin**.

The center of a flat valley is often:
- More stable
- Flatter
- Better generalizing

---

## When Polyak Averaging Works Best

> [!success]
> EMA works best when:
> - Training remains inside a **single basin**
> - The minimum is **wide and flat**
> - SGD has moderate noise (small/medium batch size)
> - Learning rate is not extremely small
> - Model is **large and overparameterized**

Large transformers and CNNs benefit strongly.

---

## When It May Fail

> [!failure]
> EMA can fail when:
> - Averaging across **disconnected basins**
> - Solutions are very far apart in parameter space
> - The minimum is extremely sharp
> - Models differ by hidden-unit permutations

Averaging unrelated runs can land in high-loss regions.

---

## Special Case: Reinforcement Learning

In Q-learning and actor-critic methods:

- Targets are bootstrapped
- Instability compounds over time

> [!important]
> EMA stabilizes target drift.
> It smooths oscillations and prevents divergence.

That is why Polyak averaging is standard in modern RL.

---

## Big Picture

> [!summary]
> Polyak averaging works not because neural networks are linear,
> but because modern loss landscapes are wide and connected.
> 
> EMA moves parameters toward high-volume, flat regions,
> which improves stability and often generalization.
