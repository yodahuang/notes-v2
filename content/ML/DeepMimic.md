---
date: 2026-06-20
pdf: "[[DeepMimic.pdf]]"
year: 2018
Arxiv: https://arxiv.org/abs/1804.02717
original title: "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills"
---
---

The following notes come from my discussion with Claude Opus 4.8

---

DeepMimic answers a narrow, concrete question: how do you get a *physically simulated* character to perform a skill — a walk, a backflip, a spinkick — that looks as good as replaying motion capture, but is driven by a controller reacting to real physics, so it can recover from a shove, adapt to a ledge, or chase a target instead of just playing back fixed frames?

## The whole thing in one breath

- **Input:** a character model (rigid-body skeleton), *one* reference motion clip $\{\hat q_t\}$ — mocap or keyframe, ~0.5–5 s long — that defines the *style*, and optionally a task reward defining a *goal*. One clip → one policy is the base unit; RL just practices that single clip exhaustively (tens of millions of samples), not a dataset of many. Handling a *set* of clips in one controller is a deliberate extension (the §7 methods below).
- **Output:** a control policy $\pi(a\mid s)$ that at 30 Hz emits PD-controller target angles for every joint; the physics engine (Bullet, 1.2 kHz) turns those into torques and integrates the body forward.
- **Training:** RL via [[PPO]]. The reward is mostly "how closely does my current pose match the reference at this point in the motion." Two non-default choices — **reference state initialization (RSI)** and **early termination (ET)** — are what make the dynamic skills trainable at all.

Everything below is detail on those bullets. The authors are explicit that no single piece is new; the *combination* — imitation reward + phase-aware policy + RSI + ET, all model-free — is the contribution, and it produces motion quality that earlier deep-RL controllers couldn't.

> [!tip] The mental model: it's per-instance, like [[NeRF]]
> The base method is **not a generalizing model** — it's per-clip optimization. One policy overfits to one clip the way a NeRF overfits to one scene; there's no transfer to a new clip, you just re-run the optimization, and the tens of millions of "samples" are the optimizer grinding on that single example. The one refinement to the analogy: it overfits to the *clip* but must still **generalize across states** — RSI and the stochastic closed loop force it to handle off-reference poses and recover from perturbations, so the product is a robust *controller* around that one motion, not memorized playback. The §7 multi-skill methods are then a "combine several per-clip experts" layer bolted on top.

## State, action, reward

**State** $s$: per-link relative position, quaternion orientation, and linear/angular velocity, all in the character's local frame (root at origin, x along facing direction), plus a **phase variable** $\phi \in [0,1]$ marking where in the clip we are — that's what tells the policy *which* reference pose it's supposed to be matching right now. A goal $g$ (if there's a task) is appended like extra state.

**Action** $a$: PD-controller *target angles*, not torques. This matters. A fixed, hand-tuned PD law handles low-level joint stabilization for free, so the policy only has to decide "what pose to aim for," not "what force to apply." It also means the policy's exploration noise becomes *smooth* motion (the PD law filters it into gradual torque) instead of jittery force injected straight into the dynamics. The gains are constants the network never sees — a property of the plant, not something learned.

**Reward** — and here's the key mechanical trick: it compares the simulated character and the reference in a **common space**. Both pose configurations are pushed through **forward kinematics** (joint angles → world-space positions of links, end-effectors, center of mass), and the comparison happens *there*. Same FK function applied to both sides; that's what makes the two comparable at all.

![[deepmimic-control-reward-loop.svg|637]]

The loop on the left is what the policy lives in: state → action → physics → new state. The two maps to keep apart are the orange box and the blue ones. **Forward dynamics** (orange) is the realized-pose physics — non-instantaneous, contact-ridden, and the reason `qₜ₊₁ ≠ āₜ`. **Forward kinematics** (blue) is the clean angles-to-positions map used only to land both poses in the same world-space for comparison. Phase φ is the synchronizer: it picks the policy's place in the loop *and* the reference frame to grade against.

> [!note] The reward, concretely (ephemeral implementation detail)
> $r_t = \omega^I r_t^I + \omega^G r_t^G$, with $\omega^I=0.7,\ \omega^G=0.3$ when a task is present. The imitation term is four exponentiated-distance pieces:
> - **pose** $r^p = \exp(-2\sum_j\lVert \hat q^j_t \ominus q^j_t\rVert^2)$, weight 0.65 — quaternion difference per joint
> - **velocity** $r^v = \exp(-0.1\sum_j\lVert \hat{\dot q}^j_t - \dot q^j_t\rVert^2)$, weight 0.1
> - **end-effector** $r^e = \exp(-40\sum_e\lVert \hat p^e_t - p^e_t\rVert^2)$, weight 0.15 — hands/feet world position
> - **center-of-mass** $r^c = \exp(-10\lVert \hat p^c_t - p^c_t\rVert^2)$, weight 0.1
>
> The exponentiation tolerates small smooth deviations gracefully — the policy is never graded on *exact* tracking, which is part of why no integral (PID) term is needed: the policy itself absorbs any steady-state bias by commanding a slightly adjusted target.

Note the careful distinction: forward *kinematics* (angles → positions) is the clean, well-behaved map used for the **reward**. Forward *dynamics* (action → actually-realized next pose, through torque + contact + integration) is the messy physics that the simulator runs — and it's exactly the thing that makes supervised learning fail.

## Why not just supervised learning / behavior cloning?

The natural instinct is: regress the policy onto the demo, maybe RL-finetune on top. It doesn't work, for two linked reasons.

**1. There are no action labels.** Mocap records *poses*, never the control signal that produced them on *this* body. The human's motor commands are unobserved and wouldn't transfer to a simulated skeleton with different mass and actuators anyway. So there's no $a_t$ to regress against — only $\hat q_t$.

**2. Hitting a target angle isn't instantaneous — time and physics are in the way.** A PD target of "0°" does not make the joint 0°. PD computes a *torque*; the engine integrates it over ~40 substeps per control step; and gravity, contact forces, and cross-joint coupling all push back. So the realized next pose $q_{t+1}$ is the output of **forward dynamics**, generally $\neq$ the commanded target. You can't supervise "which action yields the next pose" without inverting that physics — and that map is full of contact discontinuities. This is also why the tempting "differentiable FK" shortcut fails: the differentiable map you'd actually need is forward *dynamics*, not FK, and analytic gradients through contact are notoriously chaotic (the differentiable-physics line — Brax, MuJoCo MJX, DiffTaichi; cf. Suh et al. on contact gradients being worse-conditioned than RL's; SHAC for the short-horizon hybrid).

Even if you manufactured action labels via inverse dynamics, you'd hit **covariate shift**: a supervised tracker only knows the demo states, so the first tiny physics deviation drifts it off-distribution and the error compounds — worst precisely for the dynamic, contact-rich motions DeepMimic targets.

RL with an imitation reward sidesteps all of it. The reward only ever needs the *kinematic comparison* (which we have); it never needs the correct action. And because training is closed-loop — stochastic policy, imperfect contacts, RSI-perturbed starts — the policy naturally visits off-reference states and learns to *recover*, which is why these controllers survive perturbations that a BC tracker structurally cannot.

## The data

Each skill is learned from roughly **0.5–5 seconds** of reference motion. The human skills come from public mocap — the [CMU](http://mocap.cs.cmu.edu) and [SFU](http://mocap.cs.sfu.ca) databases. The T-Rex and dragon, where no mocap exists, use **artist-authored keyframe animation** — proof that the framework needs only a *kinematic reference*, not real capture.

Clips are manually processed and retargeted to each character before training: marker data → skeletal joint angles → mapped onto the simulated skeleton. The Atlas robot (169.8 kg, ~4× the humanoid's mass) is retargeted from the humanoid by simply *copying local joint rotations* — its joints were restructured into 3D spherical joints to make this trivial. But note: retargeting the *clip* is not the same as transferring the *controller* — a humanoid policy applied directly to Atlas fails completely (return 0.013), so each character is trained fresh on its retargeted clips.

## The repertoire: skills vs tasks

The whole results section is a grid of two independent axes, and the paper's `Character: Skill - Task` naming encodes it literally.

![[deepmimic-skills.png]]

**Skill** = which clip is imitated ($r^I$). Three families ([[DeepMimic.pdf#page=9|Fig. 6]]):
- **locomotion** — walk, run, crawl (plus T-Rex and Dragon walks)
- **acrobatics** — backflip, frontflip, sideflip, spin, cartwheel, roll, getup (face-up / face-down), vault
- **martial arts** — spinkick, kick, punch, baseball pitch

A skill can stand alone with no task — backflip/cartwheel are *pure imitation*, no navigation. So "backflips are hard" is a statement about **physical realizability** (coordinating angular momentum through an unsupported flight phase), not about any goal.

**Task** = an optional goal-directed reward ($r^G$) layered on a skill ([[DeepMimic.pdf#page=10|Table 3]]):
- **Target Heading** — walk/jog/run in a commanded direction
- **Strike** — hit a target with a spinkick
- **Throw** — pitch a ball to a target
- **Terrain Traversal** — mixed obstacles / dense gaps / winding balance beam / stairs, using a heightmap input

The point of the task reward is shown cleanly in Table 4: it's what bends an imitation clip toward actually *accomplishing* the goal. Strike succeeds 99% with $r^I + r^G$ but only **19%** on imitation alone; Throw is 75% vs **5%** — the clip by itself essentially never lands on a randomly-placed target.

## RSI and ET — the two ingredients the ablations credit

Both solve the same underlying problem: *getting useful reward signal for a dynamic skill before the policy can already perform it.* The ablation ([[DeepMimic.pdf#page=12|Table 5]]) shows they only matter for the hard skills — a walk trains fine without either; a backflip collapses.

- **RSI (reference state initialization)** — start each episode from a *random point along the reference clip*, not always frame 0. You can't learn a backflip's landing if you never reach a landing state to begin with; RSI just drops you there. (In RL terms it's *exploring starts*, enabled by the simulator's ability to reset to any state — a privilege a real robot lacks.) Backflip return: **0.79 with, 0.38 without**.
- **ET (early termination)** — end the episode the instant the torso or head touches the ground. Otherwise the replay buffer fills with useless "flailing on the floor" states that dominate the data and soak up network capacity — a class-imbalance problem, the same one supervised learning fights.

The pair compose: RSI gets you *into* the states worth learning from; ET keeps you *out of* the ones that aren't.

![[deepmimic-rsi-exploring-starts.svg|637]]

The strip shows the RSI half. Start every episode at frame 0 and a backflip's landing sits behind a wall of states the policy can't yet produce, so it never gets there to be rewarded; scatter the start phase across the clip and the landing becomes just another visited, gradable state.

## Combining clips into compound skills

**How prior (kinematic, no-physics) animation did it.** Earlier work stitched many clips together without ever simulating physics: motion-graph / clip-selection search [Lee et al. 2010; Safonova & Hodgins 2007], Gaussian-process latent spaces [Levine et al. 2012; Ye & Liu 2010], and learned generative models — autoencoders and phase-functioned networks [Holden et al. 2016, 2017]. Conceptually these are either discrete nearest-neighbor *retrieve-and-stitch* over a clip database, or *interpolation in a learned latent space* — but always replaying or blending recorded kinematics, never reacting to physics, which is the whole gap DeepMimic closes.

**DeepMimic's three ways** to put multiple clips into one *physical* controller (§7), from least to most explicit conditioning:

- **Multi-clip reward** — combine $k$ clips with `max`, not `sum`. (A pose can match at most one clip at a time, so a sum would optimize a blurry compromise that matches none, and would dilute the gradient — `max` makes it a "match *any* one" objective.) It's essentially unconditioned, so it only works for *similar* clips (their example: 5 walks/turns) and mode-collapses on dissimilar ones; in practice the task reward does the real clip-selecting.
- **Skill selector** — a one-hot label fed into the policy picks the clip, and the user can flip it at runtime. This is the conditional-generation option (like a class label), and the clean one — no ambiguity for the optimizer.
- **Composite policy** — train $k$ policies *independently*, then at runtime pick the next skill by a softmax over their **value functions** (whichever expert is most confident from the current physical state, $T=0.3$, no immediate repeats). Skill-to-skill transitions and automatic getup-after-a-fall emerge *for free* — it's a mixture-of-experts gate that was never trained to gate, and it's memoryless (hence the bolted-on no-repeat rule). The right formal lens is a degenerate, untrained policy-over-options.

![[deepmimic-composite-policy.svg|637]]

The trick to see in the diagram: nothing in the middle was trained to *choose*. Each expert's value function already answers "how well will things go if I take over from here?", so reading off the most confident expert at each step *is* the gate. Land flat on your back and the getup expert is suddenly the one with the highest value — recovery falls out without anyone wiring up a recovery rule.

> [!note]+ Cost, robustness, and lineage
> - **Cost:** ~30 Hz control, 1.2 kHz physics, stable PD controllers [Tan et al. 2011], TensorFlow. Training is expensive — tens to hundreds of millions of samples, ~2 days per skill — the price of the black-box, high-sample-complexity choice.
> - **Robustness / retargeting:** policies survive significant pushes and generate plausible recoveries. A landing clip captured on flat ground retrains into jumping down off a 2 m ledge; a forward-walk clip + heightmap becomes a balance-beam traversal — the policy adapts the *same* clip to environments quite different from where it was recorded.
> - **Lineage:** this is essentially the same group's **DeepLoco** plus an imitation term, but DeepLoco used *fixed* initial states and so couldn't do highly dynamic motion — exactly the gap RSI fills. **SAMCON** (Liu & Hodgins) reaches comparable acrobatic range but is a complex multi-component pipeline; DeepMimic's pitch is "a much simpler, single model-free framework."
