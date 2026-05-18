---
date: 2026-04-20
pdf: "[[dpo.pdf]]"
year: 2023
original title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
---
A good paper.  Even the "Preliminaries" part is very interesting that I feel might warrant a separate note. it's now in [[RLHF]].

---

The following note is generated from one of my discussion with Claude Sonnet 4.6

---

DPO eliminates the explicit reward model and RL loop from [[RLHF]] by reparameterizing the reward in terms of the policy itself. The key insight: the policy _is_ the reward model, via a log-ratio with the reference.

## Setup

Standard RLHF maximizes a KL-regularized reward objective:

$$
 \max_\pi \mathbb{E}_{x \sim \mathcal{D},, y \sim \pi}\left[r(x,y)\right] - \beta , D_\text{KL}[\pi | \pi_\text{ref}] 
$$

where $x$ is the prompt and $y$ is the full generated response (a complete token sequence). The KL penalty keeps the policy from drifting too far from the reference (SFT) model.

Note this is reverse [[KL divergence]] $D_{KL}(\pi \| \pi_\text{ref})$ — mode-seeking around the reference. The direction matters: the closed-form exponential-family solution below comes from the variational characterization where you minimize $D_{KL}\!\left(\pi \,\big\|\, \pi_\text{ref} \cdot \exp(r/\beta)/Z\right)$, and that decomposition requires this KL direction. Forward KL wouldn't give the same clean result.

## Derivation

### Step 1: Closed-form optimal policy

The KL-regularized objective has an analytic solution for any reward $r$:

$$
 \pi^*(y|x) = \frac{1}{Z(x)},\pi_\text{ref}(y|x),\exp\left(\frac{1}{\beta}r(x,y)\right) 
$$

where $Z(x) = \sum_y \pi_\text{ref}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)$ is the partition function.

> [!warning] $Z(x)$ is not just expensive — it's intractable
> $Z(x)$ sums over **all possible token sequences of all lengths** — a combinatorially infinite space. This is the same fundamental intractability as in energy-based models and undirected graphical models. It cannot be evaluated, period. This is why the cancellation in Step 3 is essential, not merely convenient.

### Step 2: Invert — express reward in terms of policy

Instead of reward → policy, flip it. Take logs of the optimal policy equation and rearrange for $r$:

$$
 r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x) 
$$

The reward is a log-ratio of optimal policy to reference, plus a ==term that depends only on $x$, not $y$==.

### Step 3: Plug into Bradley-Terry preference model

Human preferences are modeled as:

$$
 p(y_w \succ y_l \mid x) = \sigma!\left(r(x, y_w) - r(x, y_l)\right) 
$$

Substituting the reparameterized reward, the $\beta \log Z(x)$ terms **cancel** (same $x$, same prompt):

$$
 p(y_w \succ y_l \mid x) = \sigma!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}\right) 
$$

The cancellation depends structurally on **pairwise comparisons of completions from the same prompt**. Two pieces have to line up:

- Same prompt $x$ → same $Z(x)$ → cancels in the difference.
- Pairwise (not scalar) → the comparison takes a *difference* of rewards, which is what eliminates $Z(x)$. A single-completion scalar-reward objective leaves $Z(x)$ intact and DPO doesn't apply.

This is why BT preference data is uniquely compatible with the trick. Listwise rankings within a prompt work too (pairwise decomposition), but cross-prompt comparisons or absolute-score targets do not.


### Step 4: MLE with trainable $\pi_\theta$

Step 3 gives a _statistical model_ for preference probabilities parameterized by the policy. Replace $\pi^*$ with trainable $\pi_\theta$ and do MLE on the preference dataset:

$$
 \mathcal{L}_\text{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right] 
$$

This is binary cross-entropy. No reward model, no RL rollouts, no PPO.

> [!question] Why can we substitute $\pi_\theta$ for $\pi^*$? 
> This is just MLE — not a policy iteration argument. You have a parameterized family of distributions over preference pairs. You assume $\pi^*$ lies within (or is well-approximated by) ${\pi_\theta}$. MLE on a well-specified model recovers the true parameters. The sophistication was entirely in showing preferences can be written as policy log-ratios; the optimization step is standard.

## DPO in the discrete-sampling taxonomy

See also [[RLHF#Why Not Just Backprop? The Discrete Sampling Problem]].

Step back. The objective DPO and [[PPO]] both face is:

$$
\max_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[r(x,y)]
$$

(with KL regularization in both cases). The gradient $\nabla_\theta \mathbb{E}_{y \sim \pi_\theta}[r(x,y)]$ has the same problem as the VAE gradient $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)]$: $y$ is a discrete token sequence, so the [[Reparameterization trick]] doesn't apply. Two strategies exist:

**Fight through it.** Use the [[Score function]] estimator $\nabla \mathbb{E}[r] = \mathbb{E}[r \nabla \log \pi]$. PPO is the canonical example — [[Policy Gradient|REINFORCE]]-style gradients with clipping and importance ratios as variance reduction. Same family as Gumbel-softmax / straight-through for discrete VAE latents, like [[VQ-VAE]]: a specific trick to make the high-variance estimator workable.

**Avoid it.** Reformulate so sampling never appears in the loss. DPO is the cleanest example — the algebraic chain (closed-form optimum → reward reparameterization → BT cancellation) replaces the entire expectation $\mathbb{E}_{y \sim \pi_\theta}[\cdot]$ with log-probabilities of pre-collected completions $(y_w, y_l)$. KTO, IPO, and SimPO follow the same template with different preference assumptions.

So DPO is not "RLHF without the RL part" in some superficial sense — it's a fundamentally different strategy for the same underlying problem. PPO computes a noisy estimator of an intractable gradient; DPO algebraically transforms the problem into one where no such gradient appears.


## Key Intuitions

- The **BT model** connection is central — DPO takes it seriously as a latent variable model where the latent is the policy itself
- RLHF trained a separate reward model as an intermediate (BT regression with a scalar head), then ran PPO against it. DPO collapses both stages by exploiting the closed-form KL-constrained solution
- $\pi_\theta$ **implicitly represents the reward** through its log-ratio with $\pi_\text{ref}$

## The General Pattern

The DPO trick generalizes wherever you see:

1. A latent quantity (reward, value, energy) you don't want to model explicitly
2. A KL-regularized objective with a closed-form optimal of the form $\pi_\text{ref} \cdot \exp(\text{something}/\beta)$
3. Pairwise comparisons that let the intractable normalizer cancel

> [!note] Broader applicability
> Any exponential family model with a KL constraint has a closed-form optimal that looks like $\pi_\text{ref} \cdot \exp(\cdot)$, so wherever you see KL-regularized optimization + pairwise comparisons, DPO-style reasoning likely applies.
> 
> The same exponential-family algebra appears in the EM E-step (see [[Variational inference]]): unconstrained $q$ maximizing the [[ELBO]] gives $q^*(z) = p(z|x) \propto p(x|z)p(z)$ — same closed-form structure, different surface application. KL-regularized reward maximization is essentially variational inference with $\pi_\text{ref}$ playing the role of the prior and $r/\beta$ playing the role of $\log p(x|z)$. The intractable $Z(x)$ in DPO is the intractable evidence $p(x)$ in VI under another name.
