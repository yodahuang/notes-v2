---
date: 2026-06-29
pdf: "[[EgoScale.pdf]]"
year: 2026
Arxiv: https://arxiv.org/abs/2602.16710
original title: "EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data"
---

---
*Notes written with Claude Opus 4.8 from a discussion of the paper.*

---

EgoScale (NVIDIA GEAR, built on the GR00T N1 backbone) makes one bet: **human-to-robot transfer for dexterous manipulation is fundamentally a scaling problem, and a human hand is just another embodiment.** If you pick an action representation that humans and robots can share, you can pretrain a VLA on ~20k hours of *noisy, in-the-wild* egocentric human video, and that noisy prior beats small amounts of clean, perfectly-aligned robot data. The method itself is deliberately plain — almost everything interesting is in *what they chose to supervise* and *what the ablations justify*, which is how I'd read the paper: it's a sequence of "why this, not that" decisions.

![[egoscale-framework-overview.png]]

## The one design choice everything rests on: a shared action space

The whole thing only works because human demos and robot demos are expressed in the *same* action space. That space has two halves, and each half is a deliberate trick.

**Arm = relative wrist motion in SE(3).** Instead of absolute end-effector pose, they predict *relative* wrist transforms within an action chunk, ΔWᵗ = (W⁰_w)⁻¹ Wᵗ_w (each pose expressed relative to the chunk's first frame). This is the same move as [[Universal Manipulation Interface]] — make the trajectory relative so the absolute camera/base pose drops out.

> [!note] Same trick, slightly different payoff than UMI
> UMI uses a relative end-effector trajectory so a handheld gripper's data transfers to a robot regardless of mounting. EgoScale uses it so that **noisy in-the-wild SLAM egomotion doesn't poison the arm signal** — the camera is on a person's head moving unpredictably, and a relative formulation makes that egomotion cancel out. Same idea (relativity buys invariance), different thing being cancelled.

**Hand = 21 human keypoints retargeted to 22-DoF joint angles.** This is the part that distinguishes dexterous transfer from the gripper-based human-data line of work. Two sub-decisions worth keeping:

- **They retarget at all *because the target hand is anthropomorphic.*** The robot uses a 22-DoF Sharpa hand whose kinematics resemble a human hand, so a per-keypoint → per-joint mapping is meaningful. A parallel-jaw gripper has nothing to map fingers onto; that's precisely why gripper-based human-data methods can't do fine articulation. The human-shaped hand is what lets them keep finger detail instead of collapsing to a wrist pose.
- **Retargeting is solver-based, not a learned net** (App. D): a per-frame nonlinear program over the 22 joint angles, subject to URDF joint limits, solved with CasADi/IPOPT, warm-started from the previous frame, exp-smoothed for temporal jitter. Deterministic, optimization-in-the-loop — they trust the geometry rather than learning the map.

For human data there's no proprioceptive state, so they substitute a **learnable placeholder token** where the robot's state input would go — a small thing, but it's what lets one model ingest both human and robot data without architectural forks.

## The two ablations that actually carry the paper

Read §3 as two "why" questions. Each is settled by an ablation, and the two are about different axes.

### Why this *training strategy*? (Fig 4) — scale-of-noisy beats precision-of-aligned

Four checkpoints, five dexterous tasks (shirt-roll, card-sort, tong-transfer, bottle-cap, syringe). The numbers that matter (avg completion / success):

| Recipe | Completion | Success |
|---|---|---|
| No pretrain | 0.24 | 0.02 |
| Midtrain only (clean, aligned human+robot) | 0.53 | 0.28 |
| **Human pretrain only (noisy, in-the-wild)** | **0.71** | **0.38** |
| Human pretrain + midtrain | 0.83 | 0.56 |

The headline isn't "the recipe works" — it's the **ordering**. Noisy, unaligned, in-the-wild human pretraining *alone* beats the clean, perfectly-aligned human–robot mid-training baseline. Scale-and-diversity of messy data outweighs precision of aligned data. Then only ~4 hours of aligned robot data in mid-training is enough to cash the prior out to the top row. The mid-training stage's job is narrow: *ground* the already-learned manipulation structure into the robot's sensing/control space, not teach manipulation.

> [!note]- The three stages, concretely
> - **Stage I — pretrain.** 20,854 h egocentric human video. Mostly proprietary in-the-wild recordings (9,869 scenes, 6,015 tasks, 43,237 objects) labeled by off-the-shelf SLAM + hand-pose pipelines; plus the named 829 h **EgoDex** slice (Apple Vision Pro, accurate tracking). Supervised on relative wrist + 22-DoF retargeted joints. Full model unfrozen.
> - **Stage II — aligned mid-training.** 344 tabletop tasks, ~30 human + ~5 robot trajectories each (~50 h human, **only ~4 h robot**). Same camera rig as the robot; wrist via Vive trackers, hand via Manus gloves (25 joint transforms). VLM backbone frozen.
> - **Stage III — post-training.** Vanilla task fine-tune, 100 teleop demos/task (20 for shirt-roll). This is standard; the paper's *claim* is the Stage I+II pairing, not this.

### Why this *action space*? (Fig 8) — supervise the joints, not just the wrist

Holding everything else fixed, they swap what the human action is during pretraining: wrist-only vs fingertip-based vs full retargeted joints (completion score):

| Task | Wrist-only | Fingertip | **Retargeted joints** |
|---|---|---|---|
| Card | 0.56 | 0.17 | **0.74** |
| Tong | 0.24 | 0.76 | **0.79** |
| Bottle | 0.26 | 0.55 | **0.61** |

Wrist-only collapses on contact-timing tasks (grasps too weak, closes too early) — it never sees finger articulation. Fingertip helps but is inconsistent: small fingertip-pose errors map to implausible joint configs after IK. Retargeted joint-space is the most consistent. This is a concrete claim about *what to supervise* for dexterity, and it's the strongest "they learned something non-obvious" result in the paper.

## Cross-embodiment: the payoff of treating a hand as an embodiment (Fig 7)

If the human prior is really an *embodiment-agnostic motor prior*, it should transfer to a robot it was never shaped for. They test on the Unitree **G1** — shorter arm, different workspace, and a **7-DoF tri-finger** hand, versus the 22-DoF Sharpa it pretrained in.

Mechanism is lightweight (and the paper is thin here — they don't say much): the **VLM backbone, relative-wrist prediction, and DiT action expert stay shared**; only small **embodiment-conditioned MLP adapters** at the input (proprioception in) and output (joint actions out) are swapped per embodiment. Add a little G1 play data to mid-training and the human-pretrained policy beats a G1-only baseline (Pen-in-Bin 0.83 vs 0.45; Dish-in-Rack 0.88 vs 0.54). Crucially the G1 is *never trained from scratch* — mid-training only re-aligns an existing human-derived prior to a new body. That's the cleanest evidence for the "human = reusable embodiment" thesis.

## The scaling law (treat as suggestive, not a law)

At 1k–20k h, the offline human-action validation loss (MSE on held-out wrist+hand actions) follows a clean log-linear fit, L = 0.024 − 0.003·ln(D), R² = 0.9983, and downstream completion rises monotonically 0.30 → 0.71 with no saturation. The genuinely useful nested claim is that **offline val-loss tracks real-robot performance** — i.e. you could predict whether scaling is paying off without running expensive robot evals.

> [!warning] Read the scaling claim with the fine print
> - The fit is **5 points, 2 parameters** — R²=0.9983 is close to vacuous as a "law." And the loss only moves 0.024 → 0.015 across a 20× data increase; the offline–online correlation rides on the same 5 points.
> - **"One-shot" is one robot demo *plus* 100 aligned human demos** per task, not a single demonstration total.
> - **Internal inconsistency on Fold Shirt:** p2 says mid-training "contains only folding behaviors," p9 says Fold Shirt does *not* appear in mid-training. Likely a wording slip, but it muddies how much of the 88% is generalization vs near-distribution recall.
> - **Reproducibility ≈ 0:** the bulk ~20k h corpus is proprietary and unnamed (only the 829 h EgoDex slice is named), and the upstream hand-pose estimator + SLAM are just "off-the-shelf." The retargeting *is* fully disclosed (App. D); the data and labeler are not.

## Implementation snapshot

> [!note] Ephemeral details (GR00T N1-inherited; likely to be superseded)
> Flow-matching VLA: VLM encodes image + language instruction → DiT action expert predicts an action chunk via flow matching. Embodiment-specific MLP adapters for state-in/action-out; backbone + wrist head + DiT shared across embodiments. Training: Stage I 100k steps on 256 GB200, batch 8192, lr 5e-5, full unfreeze → Stage II 50k steps, batch 2048, VLM frozen → Stage III 10k steps, batch 512. Robot: Galaxea R1Pro, bimanual 7-DoF arms in relative EE space, 22-DoF Sharpa Wave hands, head + two wrist RGB cameras.
