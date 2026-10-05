---
date: 2026-10-04
---

---
Notes written with Claude Code (Claude Opus 5.5) from a reading discussion.

---
Papers keep saying "flow / diffusion policies are hard for RL" and "Gaussian policies are bad", which makes it sound like nothing works. Both statements are true, because they answer **two different questions**:

1. **Can the policy class represent the behavior data?** Correlated action chunks, several distinct modes.
2. **Does it give RL something to work with?** A log-probability, or an action that is a cheap differentiable function of the weights.

| | represents the data | RL handles |
|---|---|---|
| Gaussian (diagonal) | poorly: unimodal, independent dims | both, cheap |
| ACT-style (CVAE, L1, latent = prior mean at test time) | partly: correlated chunks, but one mode | pathwise only (deterministic); add Gaussian noise for a likelihood, and you're back to row 1 |
| Flow / diffusion | well | neither cheaply |

So switching from an ACT-style to a flow policy fixes question 1 and breaks question 2. Every method below is a way to keep the flow's answer to 1 while getting around 2.

## Question 1: representing the data

Two separate failures, often blurred together:

- **Correlation within a chunk.** A 5×7 action chunk's entries are strongly coupled: step 3 should look like step 2. Independent per-dimension Gaussian noise wiggles each one separately, which is jitter, not exploration. A full covariance (Cholesky factor, ~630 numbers for 35 dims) or temporally correlated noise (DDPG's Ornstein–Uhlenbeck noise) fixes this.
- **Multimodality.** If the data is "LLLLL or RRRRR", any Gaussian, with any covariance, is still one blob. The best fit sits in the middle, which may be the one thing nobody did. A covariance can't fix this; an expressive generator can.

This matters for RL beyond imitation quality. The behavior constraint and the exploration both inherit from this model. A model that blurs the modes constrains the policy toward behavior that isn't in the data and explores incoherently. That's why the Gaussian chunk actor fails in [[Q-chunking]].

## Question 2: what RL needs from a policy

There are two standard ways to improve a policy, each needing one handle:

- **Likelihood:** $\log \pi(a \mid s)$. Used by [[Policy Gradient]], [[PPO]] ratios, SAC's entropy term, and any explicit KL to a reference policy.
- **Pathwise gradient:** $a = g_\theta(s, \text{noise})$ cheap and differentiable, so $\nabla_\theta Q(s, a) = \nabla_a Q \cdot \nabla_\theta a$. This is how [[Q learning#Use a NN to tell what's the max|DDPG]], TD3, and SAC's reparameterized actor learn.

A flow model has neither cheaply:

- **Likelihood:** the sampler defines a real distribution, but $\log \pi(a \mid s)$ requires integrating the divergence of the velocity field along the ODE path. That is expensive and noisy.
- **Pathwise:** the action is the end of a 10-step ODE, so $\nabla_\theta a$ means backpropagating through every step, which is costly and unstable like backprop through time.

Note that "it's trained by regression" is *not* the problem. ACT trains by regression too. What matters is what the trained model exposes.

## A third, separate issue: the constraint needs a handle too

Independently of the model class, offline-to-online RL has to keep the policy near the data, because $Q$ is extrapolating everywhere else ([[On and off policy Learning#Why one-step TD works off-policy|coverage]]). The usual constraint is a KL to the behavior policy, which needs log-probs: question 2 again. So with flows the *constraint* has to be rebuilt as well, not just the improvement step.

## The bridges

Each method keeps the flow and replaces whichever handle it can't get:

| Method                                                   | Replaces   | How                                                                                                                                  |
| -------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| [[Best-of-N sampling]] (QC in [[Q-chunking]])            | both       | needs only samples; the constraint comes free as a KL bound                                                                          |
| One-step distilled actor (FQL, QC-FQL)                   | pathwise   | train a single-pass actor near the flow; the constraint becomes $W_2$ via shared noise                                               |
| Advantage-weighted flow matching (AWR-style)             | likelihood | weight the regression loss by $\exp(A/\lambda)$ instead of the log-likelihood                                                        |
| Guidance / conditioning ([[CFGRL]], RECAP in [[Pi 0.6]]) | both       | improve at sampling time by contrasting conditioned and unconditioned velocity                                                       |
| RL in the noise space (DSRL)                             | both       | keep the flow fixed as a decoder; run ordinary Gaussian RL on its input noise $z$, so every perturbation decodes to a coherent chunk |

The last row also answers the covariance problem from question 1: noise in $z$ is already shaped by a model that knows the correlations and the modes.
