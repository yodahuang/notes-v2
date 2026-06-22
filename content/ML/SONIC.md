---
date: 2026-06-21
pdf: "[[sonic.pdf]]"
year: 2026
Arxiv: https://arxiv.org/abs/2511.07820
original title: "SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control"
---

---
*Notes written by Claude Opus 4.8 from a reading discussion.*

---

SONIC's one-sentence pitch: **take [[DeepMimic]]-style motion tracking, stop training one policy per clip, and scale it like a foundation model** onto a real Unitree G1. The three scaling axes: model **1.2M → 42M params**, data **~100 M frames (≈700 h of mocap — same dataset, two units)**, compute **2k → 21k GPU-hours**. The empirical claim is that *motion tracking* is the right "scalable task" for humanoid control: it gives dense per-frame supervision from mocap with no per-task reward engineering, and it scales favorably (performance keeps climbing with data/compute, and policies generalize to unseen motions).

Worth flagging up front: the "supersizing" is **modest by LLM standards** — 42M params is tiny. It's supersized *relative to prior humanoid controllers*, which were typically 3-layer MLPs on a handful of GPUs. The contribution is showing this task *scales at all*, not the absolute size.

Almost nothing here is a new *algorithm*. The tracker is DeepMimic / BeyondMimic. The planner is [[MaskGIT]]-family masked generation. The token is FSQ. What's genuinely new is the **universal token as a source-agnostic goal interface** — one tracker that doesn't care whether its target came from a mocap clip, a planner, a VR rig, a text-to-motion model, or a VLA — plus the demonstration that all of this composes at scale on hardware. It reads as a *systems + scaling* paper, not an algorithmic one.

> [!tip] This paper should have been three papers
> It crams a tracker, a planner, and a whole integration-and-results story into one dense writeup with poor figures, and quietly reuses "tracking" for two different things. These notes split it the way it should have been written, and the diagrams below are reconstructions, not from the paper.

> [!info]- Timeless idea vs. ephemeral implementation
> The durable ideas: *motion tracking as a scalable, densely-supervised foundational task*; *a discrete token as a modality-agnostic interface between any motion source and one low-level controller*; *match the latent's structure to whatever consumes it*. The ephemeral parts (will date fast): the specific net sizes, FSQ-32-32, the G1 joint count, the exact success numbers, GR00T N1.5 as the VLA.

# Part 1 — The tracker: what is actually tracking what

**The tracking problem, stated plainly:** the tracker is a *low-level controller* that, given a stream of target poses (the "motion command"), makes the physical robot reproduce them frame by frame — handling balance, contact, and recovery itself. It does not generate motion; it *chases* a motion it's handed. "Tracking" just means imitating that target under physics.

What makes the paper confusing is that it overloads the word and never says cleanly *what is tracking what*. There are really **two stacked models, two learning paradigms**:

![[sonic_system_overview.svg|680]]
![[sonic_tracking_og_overview.png]]

1. **High level** — a *goal source* produces a kinematic motion target $g$ (where joints/root should be over the next short window). This can be a recorded clip, the planner (Part 2), a VR stream, GEM (text/video/music), or a VLA. None of these reason about physics.
2. **Low level** — the **Universal Control Policy**: a goal-conditioned policy $\pi(a \mid s^p_t, g)$ that, every 50 Hz tick, takes proprioception $s^p_t$ plus the target and emits a single 29-dim PD joint command. This *is* DeepMimic, just scaled and made source-agnostic. It handles balance, contact, recovery.

So "tracking" = the low-level policy imitating whatever target it's handed, frame by frame. The catch the paper never says out loud: in Sec 2.1 that target is a held-out mocap clip (ground truth), but in interactive control it's a trajectory the planner *hallucinated one frame ago* — mechanically identical to the tracker, only the source differs. This is exactly goal-conditioned hierarchical RL, except the high level is trained generatively rather than as an RL policy.

### How it's trained: PPO, then what the losses are

The tracker is trained with **[[PPO]]** — RL in sim (Isaac Lab), not behaviour cloning. The full objective is just two families added together:

$$\underbrace{\mathcal{L}_{ppo}}_{\text{DeepMimic imitation reward}} \;+\; \underbrace{\mathcal{L}_{recon} + \mathcal{L}_{token} + \mathcal{L}_{cycle}}_{\text{token-space geometry}}$$

The first term, the **PPO reward**, *is* the [[DeepMimic]] imitation reward: a dense, per-frame, per-link match against the target — exponentiated squared error on root position/orientation, the 14 body-link positions/orientations/velocities, and end-effector positions (heaviest term). All of the behavioral correctness lives here.

The second family — reconstruction, token-alignment, cycle — only shapes the **token space**; it never touches the robot's simulated joint states (the robot could fall over and these stay zero). So the whole thing is just **DeepMimic's reward, plus some token-bookkeeping losses** — and the next two diagrams are entirely about that second family.

### The universal token, and why discrete

Each source goes through a thin per-modality encoder ($\mathcal{E}_r$ robot, $\mathcal{E}_h$ human/SMPL, $\mathcal{E}_m$ hybrid) into one shared latent, quantized by **FSQ** (2 tokens × 32 dims = 64) into token $z$, which $\mathcal{D}_c$ decodes to the action. The encoder-decoder exists so **one** control decoder is agnostic to where the target came from — adding a new source later costs a new encoder, not a new policy.

Why quantize at all? Because the eventual consumer is a VLA, and that's a discrete-by-nature predictor. This is the mirror image of why [[Latent Diffusion Models]] / [[Stable Diffusion 3]] went *continuous* (KL-VAE) — their consumer (diffusion) is a continuous density estimator, so a [[VQGAN|VQ]] codebook would only add a quality ceiling. SONIC's consumer is a language-model-style token predictor, so discreteness turns a brittle 64-d continuous regression into an easy classification. Table 3: a VLA predicting tokens vs. raw SMPL = **68% vs 27%** average success (60% vs 0% on the hardest task). FSQ over VQ-VAE specifically to dodge codebook collapse on a hugely diverse motion distribution, and for clean straight-through grads under [[PPO]] (Table 4a: +8.7 mm MPJPE-L over a comparable VQ-VAE).

> The general principle worth keeping: **match the latent's structure to whatever model consumes it.**

### The three losses and the supervision split

The "aha" that makes the system coherent: PPO and the aux losses supervise *different decoder heads*, but share the trunk.

![[sonic_supervision_split.svg|680]]

- $\mathcal{D}_c$ (control decoder) → single 29-dim frame, supervised **only** by $\mathcal{L}_{ppo}$ (reward). It's the only thing deployed.
- $\mathcal{D}_r$ (motion decoder) → reconstructs the full **10-frame chunk** $\hat g_r$ from the token *alone* (no proprioception), supervised by the three aux losses. Training-only; delete it post-training and deployment is unaffected. Token-only input is deliberate — it forces $z$ to be a self-contained "intent" a VLA can later predict.
- **Encoders + quantizer get gradient from both.** That coupling is the whole point: RL shapes $z$ for control while the aux losses keep it well-structured (aligned across modalities, round-trip consistent).

$\mathcal{L}_{recon}$ vs $\mathcal{L}_{cycle}$ tripped me up — they live in different spaces:

![[sonic_recon_vs_cycle_loss.svg|680]]

$\mathcal{L}_{recon}$ compares $\hat g_r$ to $g_r$ in **pose space** (only checks the aux output looks right). $\mathcal{L}_{cycle}$ re-encodes $\hat g_r$ and compares **in token space** ($\mathcal{E}_r(\mathcal{D}_r(z_h)) \approx z_r$) — policing the representation that's actually load-bearing, since two poses close in MSE can land in different FSQ bins. This is **not** CycleGAN: there's full paired supervision (synchronized $g_r,g_h,g_m$ from the same retarget), so cycle-consistency isn't standing in for missing pairs — it's enforcing something pose-MSE structurally can't.

> [!note] Worth remembering for re-reads
> - **Retargeting matters** (unlike DeepMimic, whose character was built to match the skeleton): G1 has a real morphology gap, so GMR + PyRoki (IK) map mocap → G1 joints, then infeasible motions are filtered.
> - **Why two tokens?** Genuinely unexplained. Every quantizer ablation fixes the count at 2 and only sweeps per-token capacity. The 64-d VLA action is just downstream bookkeeping, not a justification.
> - **The "CLIP mean-seeking" worry** ($\mathcal{L}_{token}$ is MSE alignment — why no blurry average?): because the $g_h \to g_r$ correspondence is near-deterministic (one retarget per clip), not multi-modal like image generation; FSQ then snaps small averaging errors back into the right bin; and it's *embedding alignment*, not generation. The worry **does** resurface at the VLA, where one observation can admit multiple valid actions — and the paper never analyzes whether failures there are multi-modality.

# Part 2 — The planner: MaskGIT inpainting in motion-token space

**What the planner does:** it's the high-level goal source — a generative model that *invents* a short kinematic motion for the tracker to chase. **Purely kinematic — no physics, no balance.** Given a context window (recent robot poses) and a target keyframe, it hallucinates a plausible 0.8–2.4 s pose trajectory between them and hands it to the tracker. (Note: its token sequence $\{z_t\}$ is *not* the tracker's control token $z$ — same letter, unrelated space.)

### It's image inpainting, in 1-D and temporal

Structurally it's [[MaskGIT]]/MoMask: encode the segment to $T/4$ discrete tokens, clamp the start/end keyframes as "known," mask everything between, and iteratively unmask by confidence under a cosine schedule (slow start, fast finish).

![[sonic_planner_maskgit_iterations.svg|680]]

The detail that causes most confusion: **all tokens start masked**, including boundary positions — the keyframes enter only as a raw side-channel into the network, never as pre-seeded tokens. Training is **single-pass**, not unrolled: mask a uniformly-random fraction [0,100%] and predict the rest (BERT-style). That generalizes to multi-round inference because every intermediate "k finalized, rest masked" state looks like a training example at that mask ratio. ($L_{max}$ never stated; MaskGIT's own sweet spot is ~8–12.)

> [!info]- Lineage I didn't know was a named line of work
> MaskGIT (Chang 2022) is the ancestor — confidence-based parallel decoding, ~8 steps instead of 256, cosine schedule beat linear in their ablations, and it natively does inpainting via clamp/mask. SONIC cites only descendants (MoMask, MMM, MAGVIT, Open-MAGVIT2), not MaskGIT itself. The clean mental model given a diffusion background: this **is** discrete-state diffusion (D3PM absorbing-state) — masking = noise, unmasking = denoising, cosine mask schedule = noise schedule.

### From command to keyframe: the spring model + three paths

A controller command never conditions the network directly — its only job is to build the *end keyframe*. For navigation, a critically-damped **spring model** converts a velocity/direction command into a target root position+heading at $t{+}1\text{s}$ (purely geometric; mostly a smoother that prevents degenerate keyframes like abrupt reversals). The three keyframe-construction paths are genuinely different and the paper buries them in one paragraph:

![[sonic_keyframe_module.svg|680]]

The unifying point the paper never states: the end keyframe is *deliberately underspecified* — one coarse pose. The system works **because** the planner is asked only to bridge two sparse constraints from its learned distribution, not to replay an authored motion. Hence "a single clip per skill" suffices, and new apps need no retraining of planner or tracker.

### "Which foot goes first?" — multimodality, resolved two ways

A nice stress-test. The **tracker** never faces it: its reference already specifies which foot lifts when, so there's no average to collapse. The **planner** does face it (training data has both left- and right-first starts), and the **discrete token space** is the fix — its softmax puts mass on *both* discrete options and samples one valid mode, instead of averaging to an invalid in-between (same reason an LM handles ambiguous words). Start-keyframe asymmetries usually break the tie in practice. Where it genuinely persists: truly symmetric transitions (idle → boxing) where the choice is effectively random — fine for the tracker, but potentially inconsistent supervision for a VLA. Untested.

# Part 3 — Integration, scale, and what it leans on

**What this part is:** everything that sits *on top* of the tracker once you have it — driving it with a VLA for autonomous tasks — plus the evidence that the central bet (scaling tracking) actually pays off.

### The scaling result is the actual contribution

![[sonic-scaling.png|680]]

Not obvious a priori, and it's the paper's real empirical claim: success rate and MPJPE-L improve steadily with **data, model size, and compute**, with gains most pronounced on out-of-distribution (test-content) motions — i.e. scale buys generalization. Largest model 99.6% / 23.8 mm on 182 unseen sub-categories; smallest 98.0% / 27.7 mm.

### VLA integration (the gap the paper scatters across four sections)

Two modes, never cleanly distinguished: (a) **whole-body** — the VLA (GR00T N1.5) predicts the 78-dim action (64-d token + 14-d hands) and *streams the token directly* into $\mathcal{D}_c$, bypassing all encoders via a "token-only" protocol; (b) **3-point** — the VLA emits teleop-format signals and rides the same planner+$\mathcal{E}_m$ path a human operator would. The elegant bit: VLA training data is collected by recording *(camera obs, active token $z$)* pairs during teleoperation, so the VLA simply learns to speak the same token language the human's motion produced. The seam is the token space; the bet is that a robust-enough tracker + in-distribution VLA tokens compose at deploy time (75% avg success, but no failure decomposition into VLA-error vs tracker-error vs distribution-mismatch).

> [!caution] What it leans on / where the paper is thin
> - **Not end-to-end.** Retargeting (GMR+PyRoki), planner (masked-gen), tracker (PPO), GEM, and the VLA are all trained separately and never share gradients. The token space, keyframe module, and alignment losses are *engineered handoffs* precisely because the pieces can't be trained jointly. "Interesting engineering choices," not one learned system.
> - **Planner training is ~one sentence** ("same data, mask ratio uniform [0,100%]") — no loss, size, or GPU-hours given.
> - **Ablations underwhelm:** consistency losses removed only as a bundle (can't separate $\mathcal{L}_{token}$ from $\mathcal{L}_{cycle}$); token count never ablated; no new-motion-coverage test; no planner mode-selection study.
> - **Adaptation (inferred, not shown):** new robot ⇒ effectively full retrain (reuse mocap, re-run retargeting, relearn token space); new action ⇒ tracker often zero-shot (99.6% on unseen sub-categories, but floor-contact/extreme poses like crawls fail), planner maybe just a library add, VLA needs fresh demos.
