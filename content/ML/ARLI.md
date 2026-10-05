---
date: 2026-10-04
pdf: "[[ARLI.pdf]]"
year: 2026
Arxiv: https://arxiv.org/abs/2608.23831
original title: "Learning to Act While Waiting: RL Finetuning of Generalist Robot Policies Under Inference Latency"
---

---
Notes written with Claude Code (Claude Opus 5.5) from a reading discussion.

---

A big VLA takes $d$ control steps to produce an action chunk, so in deployment you run it **asynchronously**: start computing the next chunk while the current one is still playing. For RL this is a problem, because the policy decides at $t-d$ but its actions take effect at $t$. Conditioned only on $s_{t-d}$, the RL state is no longer Markov, and plain RL finetuning stops learning entirely. ARLI (Asynchronous RL with Intermediate Information) fixes this by giving the RL policy more inputs. And yes, it really is about that simple:

- **Setup.** A frozen diffusion/flow VLA $\pi_{PT}$ is finetuned with [DSRL](https://arxiv.org/abs/2506.15799): a small policy $\pi_{RL}$ picks the input noise $w$, and $\pi_{PT}$ denoises it into a chunk $A_t$. Only $\pi_{RL}$ and its critic are trained, online, from task reward.
- **The change.** Instead of $\pi_{RL}(s_{t-d})$, use
$$
s^{RL}_t = \big(\, s_{t-d},\;\; a_{t-d:t-1},\;\; s_{t-d_{RL}} \,\big)
$$
  which is the stale observation, the **committed actions** that will play out while inference runs, and a **fresher mid-inference observation** $s_{\text{mid}} = s_{t-d_{RL}}$.
- **Inference.** The VLM backbone starts on $s_{t-d}$. Only at $t-d_{RL}$, just before the action expert needs its noise, does $\pi_{RL}$ run on $s^{RL}_t$ and output $w$. The action expert turns $(h, w)$ into $A_t$, which begins executing at $t$ from index $d$. RTC inpainting stays on as well.

![[arli-timeline.svg|680]]

## Async inference breaks the Markov state

Take synchronous inference first: pause, think, act. You get stop-and-go motion, and with $\pi_{0.5}$ on real hardware the paper finds that both the base policy and RL do poorly in this mode. Asynchronous inference ([[Real Time Chunking|RTC]]-style) hides the latency, but the chunk you are generating is built from a state $d$ steps old, while it will be *executed* from wherever the robot has got to by then.

For naive async DSRL this means $\pi_{RL}$ chooses $w$ from $s_{t-d}$, but the reward it gets depends on $s_t$, and $s_t$ depends on actions that $s_{t-d}$ says nothing about. The same $(s, w)$ can lead to very different outcomes, so the critic can't fit it. In the paper's experiments, naive async DSRL fails or learns very slowly.

## The Markov state is the robot plus the pending command queue

This is the standard delayed-control trick: **state of a delayed system = physical state + the control pipeline.** At decision time the actions $a_{t-d:t-1}$ are not choices any more. They are the tail of the previous chunk, already committed and about to execute. Together with $s_{t-d}$, they let $\pi_{RL}$ predict (roughly) where the robot will be at $t$, when its own chunk takes over.

> [!important] Committed, not candidate
> The actions added to the state are the ones **already locked in** during the inference window. They are not the action being chosen now. Getting this right is what makes the augmented state (approximately) Markov again.

What this misses is anything stochastic during the window: a moved object, a slip, a disturbance. $s_{t-d}$ plus the commands can't predict those. That is the job of the second input.

## Late-binding: a fresher observation, without adding latency

Most VLAs split into a **VLM backbone** $\pi^{vlm}_{PT}$ (images + language → embeddings $h$) and an **action expert** $\pi^{ae}_{PT}$ (the denoiser). The backbone takes most of the time: about 2/3 of a $\pi_0$ call. DSRL only touches the *noise input*, and the action expert doesn't need that until the backbone is done. So $\pi_{RL}$ can be called late, at $t-d_{RL}$ (where $d_{RL}$ covers $\pi_{RL}$ + action expert), and read an observation that the VLM never saw:

$$
h = \pi^{vlm}_{PT}(s_{t-d}), \qquad w \sim \pi_{RL}(s_{t-d},\, a_{t-d:t-1},\, s_{t-d_{RL}}), \qquad A_t = \pi^{ae}_{PT}(h, w).
$$

The VLA itself isn't any faster. What changes is that the part of the system that **makes the RL decision** works from more recent information, with no extra latency. The semantic understanding is still stale, and the steering is fresh. This is a systems trick and needs a VLA with this split. Without it you still get the committed-actions half of ARLI.

## Lineage: RTC's insight, reused as information

ARLI is closely related to [[Real Time Chunking|RTC]] and especially to *training-time* RTC ([Black et al. 2025](https://arxiv.org/abs/2512.05964)). All of them start from the same observation: "while I'm generating, a known prefix of actions will play, so I should take it into account." What differs is **what the committed actions are used for**:

![[rtc-vs-arli.svg|680]]

- **RTC** uses them as a **continuity constraint**. They are inpainted into the denoising so that the new chunk joins the old one smoothly. The state information is still only $s_{t-d}$.
- **Training-time RTC** retrains the policy so it is conditioned on the prefix.
- **ARLI** leaves $\pi_{PT}$ frozen and passes the prefix as **input to the small RL policy**. Since $\pi_{RL}$ is trained from scratch anyway, conditioning it on extra inputs costs nothing, while getting a frozen VLA to accept new conditioning would. $\pi_{RL}$ can then learn any reward-improving use of that information, not just smoothness.

The two combine without trouble. ARLI keeps RTC: DSRL picks the initial noise, and RTC's guidance still runs during denoising. RTC mainly makes learning faster and more reliable. ARLI without RTC often works too.

## Theory is chunked Q-learning; the implementation is DSRL-SAC

Two separate layers here, which are easy to mix up:

- **Theory.** It builds on the chunk-level Q-function framework (Decoupled Q-chunking, Li et al., which extends [[Q-chunking]]). A delayed policy keeps the committed prefix fixed and optimizes only the rest of the chunk with $Q_{ac}$. The paper defines a *delayed oracle optimality gap* $\omega_d$: how much you lose by fixing the next chunk $d$ steps early. Its Proposition 1 then shows that ordinary Q-learning on delayed observations gets within $\frac{1}{1-\gamma^{k-d}}\,\omega_d$ of the undelayed optimal chunk policy, assuming the data covers enough. In short, delay costs you only as much as it costs an oracle, and the fresher $s_{\text{mid}}$ makes $\omega_d$ smaller.
- **Implementation.** DSRL-SAC with a *single repeated latent*. The actor outputs $w_{\text{single}}$, which is tiled across the chunk axis. The critic is $Q(s^{RL}, w_{\text{single}})$, and one generate-and-execute cycle of a chunk counts as one latent-MDP step. Intra-chunk observations are ignored. So the RL action is a **noise vector**, not a robot action chunk.

The paper never says outright why it used DSRL rather than a chunk-Q learner. It does say DSRL improves faster than residual RL in this setting, and DSRL fits the "freeze the VLA, bolt on a small learner" story. The reusable idea is the state augmentation. It would carry over directly to a chunk-Q method: score $Q(s_{t-d}, s_{\text{mid}}, a_{\text{committed}}, A_{\text{candidate}})$, and optimize/sample only the part of the chunk that isn't committed.

> [!note]+ Implementation details (snapshot)
> - **Inputs.** Committed actions are flattened and concatenated with the state vector before the MLP. $s_{\text{mid}}$ images are channel-concatenated with the $s_{t-d}$ images, through a small 4-layer CNN at 64×64.
> - **Sim.** Kinetix (mjc swimmer, walker, car launch) with RTC's flow checkpoints at $d=4$, $d_{RL}=1$. AlohaTransferCube with $\pi_0$ (3.3B) at $d=20$, $d_{RL}=10$.
> - **Real.** Bimanual UR5e, $\pi_{0.5}$ LoRA-finetuned on 60 Hz demos, chunk 50, execute horizon 20, $d=10$, $d_{RL}=7$ on an RTX 5090. With RTC the action expert gets slower, and the VLM is only about 40% of inference time.
> - **Residual RL baseline.** A per-step MLP on the *current* $s_t$, so it has no latency problem. It still loses to ARLI, because DSRL's steering inside the denoiser beats a small bounded residual ($\le 0.1$).

## What the results lean on

- **Real world.** From about 40% base success, ARLI reaches about 100% on Assembly, Shoe-in-Bag and Bag-Placement in 100–125 episodes. Sync DSRL and async DSRL+RTC stay under about 80% (or need about 2× as many episodes on Shoe-in-Bag). Committed actions alone match full ARLI except on Bag-Placement, where it fails to converge, so how much $s_{\text{mid}}$ matters depends on the task.
- **Ablations (sim).** Both inputs together are clearly better than either alone. ARLI degrades more slowly than DSRL as $d$ grows. A smaller $d_{RL}$ (fresher) is better, and the big jump comes when $d_{RL} < d/2$.
- **Learned behavior.** On held-out Shoe-in-Bag replays, later checkpoints move the noise mean away from zero and shrink its variance, mostly in the alignment/insertion phase. So it reshapes the noise depending on the phase of the task, rather than rescaling it globally.
- **Limitations (as stated).** The late-binding trick needs a VLM/action-expert split where only the expert takes noise. RTC may *hurt* reactivity, because it forces old actions to play out. And everything inherits DSRL's weaknesses: tasks where DSRL struggles will be hard for ARLI too.
