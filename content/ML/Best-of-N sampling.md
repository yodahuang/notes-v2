---
date: 2026-10-04
aliases:
  - Best-of-N
  - BoN
---

---
Notes written with Claude Code (Claude Opus 5.5) from a reading discussion.

---
Draw $N$ samples from a reference distribution $p$, score them with some function $f$, keep the winner:

$$
a_1, \dots, a_N \overset{iid}{\sim} p, \qquad a^\star = \arg\max_i f(a_i)
$$

The point worth remembering: **the distribution of $a^\star$ is itself a distribution**. Call it $q_N$. In RL with $p$ a behavior policy and $f = Q$, $q_N$ is a policy, and an improved one. In LLMs with $p$ the SFT model and $f$ a reward model, it's the baseline that RLHF is often compared against.

## Why it can't drift far from $p$

Every candidate came from $p$, so selection can only *reweight* $p$; it can't put mass where $p$ has none. Ignoring ties, $a$ wins when the other $N-1$ draws score lower, so

$$
q_N(a) = N\, p(a)\, F\big(f(a)\big)^{N-1}, \qquad F(v) = \Pr_{a' \sim p}\big[f(a') \le v\big]
$$

and the density ratio is capped: $q_N(a)/p(a) \le N$. The tighter known bound is

$$
D_{\mathrm{KL}}(q_N \,\|\, p) \le \log N - \frac{N-1}{N}
$$

Notice what it doesn't depend on: **$f$**. You can swap the scorer every iteration (a $Q$-function being trained, say) and the bound still holds, as long as the procedure stays "sample $N$ from $p$, pick one". $N$ is the knob: $N = 1$ is just $p$; larger $N$ is more aggressive optimization against $f$, and more exposure to $f$'s errors.

The bound is distributional. It doesn't say any single $a^\star$ is close to a typical sample of $p$, only that the *distribution* of winners isn't far from $p$.

## As a policy-improvement operator

In offline RL this gives a behavior constraint for free: $p$ is a BC policy, $f$ is $Q$, and $q_N$ is "improve on the data, but stay within its support". It also solves the continuous $\max_a Q(s,a)$ problem from [[Q learning#Continuous Actions]] by search instead of by an actor network.

Same family as other "reference distribution + value preference" operators:
- **AWR**: reweight dataset actions by $\exp(A/\lambda)$ and *fit* a new policy to them.
- **[[CFGRL]]**: tilt the generative sampling dynamics with guidance.
- **Best-of-N**: reweight by rank, and *don't fit anything*. The selection runs every time you act, so you pay $N$ samples plus $N$ scores per decision. Training an actor to reproduce the improvement is the fix; see QC vs QC-FQL in [[Q-chunking]], and the other workarounds in [[RL with generative policies#The bridges]].
