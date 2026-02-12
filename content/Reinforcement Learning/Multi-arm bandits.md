---
date: 2025-10-18
---

This about Exploration vs Exploitation, and we focus on multi-armed bandits (MABs). They are different from "Full RL" in the following sense:
- There's only one state in MABs. The action taken does not change the state. 
- The goal is *online learning* to minimize regret (note this is the same thing as maximizing reward). Learning and evaluation are the same thing. You pay a "price" for your exploration.
- In contrast, in full RL, we expect the agent to be terrible and make tons of mistakes (i.e., have massive regret) for millions of steps. This is just the "cost of training." The final metric is the total reward of the finished, frozen policy when run on new, unseen test scenarios.

Some analogies:

You've moved to a new city and you're trying to find the best restaurant for your daily lunch.
- Every day you _don't_ eat at the _actual_ best restaurant, you've incurred "regret" (a sub-optimal lunch).
- You can't "train" for a year by eating at 1,000 restaurants and _then_ "deploy" your lunch policy. The act of "training" (trying a new place) _is_ your life. Your goal is to have the best possible set of lunches _starting from today_.
Other classic example involves showing $k$ different ads to a user (boring), clinical trials, dynamic pricing.

## Why don't I see it in general RL
Seems like this is a good way to do things if I want to balance exploration vs exploitation with no state state information. Why are we still using $\epsilon$-greedy?
The answer is that methods like UCB doesn't scale well. Or, they are not able to handle massive or continuous state / action space. For more, see [[#Appendix why UCB doesn't scale]].

## Back to definition
The **action value** for action $a$ is the expected reward

$$
q(a) = \mathbb{E}[R_t | A_t = a]
$$

The **optimal value** is

$$
v_* = \max_{a \in \mathcal{A}} q(a) = \max_a \mathbb{E}[R_t | A_t = a]
$$

**Regret** of an action $a$ is

$$
\Delta_a = v_* - q(a)
$$

Note there is only one state, and there exist such an action that is ALWAYS the best. 

We want to minimize the total regret:

$$
L_t=\sum_{n=1}^tv_*-q(A_n)=\sum_{n=1}^t\Delta_{A_n}
$$

## Greedy and epsilon greedy
Greedy is just greedy.
The **$\epsilon$-greedy** algorithm:
* With probability $1 - \epsilon$ select greedy action: $a = \underset{a \in \mathcal{A}}{\text{argmax}} \; Q_t(a)$
* With probability $\epsilon$ select a random action
* Equivalently:

$$
\pi_t(a) = \begin{cases} (1 - \epsilon) + \epsilon / |\mathcal{A}| & \text{if } Q_t(a) = \max_b Q_t(b) \\ \epsilon / |\mathcal{A}| & \text{otherwise} \end{cases}
$$

## [[Policy Gradient]]
We want to maximize total reward. We can do it by gradient ascent: in each step we make it better. Think about it this way: the total expected reward is a function of $\theta$ and picture in your head gradient ascent (since it's reward, not loss).

$$
\theta_{t+1}=\theta_t+\alpha\nabla_\theta\mathbb{E}[R_t|\pi_{\theta_t}]
$$

This expected reward is not the "reward this time" since the policy can be stochastic. How can we compute the gradient? While we can use sample based methods to approximate expectation, here we still need to sample $R_t|\pi_{\theta_1}$, which is not known.

Here comes the log-likelihood trick (also known as [[Policy Gradient|REINFORCE]] trick)

$$
\begin{aligned}
\nabla_\theta \mathbb{E}[R_t | \pi_\theta] &= \nabla_\theta \sum_a \pi_\theta(a) \overbrace{\mathbb{E}[R_t | A_t = a]}^{= q(a)} \\
&= \sum_a q(a) \nabla_\theta \pi_\theta(a) \\
&= \sum_a q(a) \frac{\pi_\theta(a)}{\pi_\theta(a)} \nabla_\theta \pi_\theta(a) \\
&= \sum_a \pi_\theta(a) q(a) \frac{\nabla_\theta \pi_\theta(a)}{\pi_\theta(a)} \\
&= \mathbb{E} \left[ R_t \frac{\nabla_\theta \pi_\theta(A_t)}{\pi_\theta(A_t)} \right] \quad = \mathbb{E} [R_t \nabla_\theta \log \pi_\theta(A_t)]
\end{aligned}
$$

- In the first step we are expanding that expectation into and show $\pi$ is just probability of $a$. Note $a$ is introduced.
- And the $q$ value doesn't depend on $\theta$, this is great.
- Some tricks so the whole thing is an expectation again by introducing back $\pi_{\theta}(a)$.

Hey hey, what changed? It goes from the gradient of theta w.r.t. an expectation to the expectation of a gradient, and that we can sample.

$$
\theta=\theta+\alpha R_t\nabla_\theta\log\pi_\theta(A_t)
$$

### Baseline
For any $b$,

$$
\begin{aligned}\sum_ab\nabla_\theta\pi_\theta(a)&=b\nabla_\theta\sum_a\pi_\theta(a)\\&=0\end{aligned}
$$

This means we can subtract a **baseline**, and instead use

$$
\theta = \theta + \alpha(R_t - b)\nabla_\theta \log \pi_\theta(A_t)
$$

Baselines *do not* change the expected update, but they *do* change variance

## UCB

Regret grows at least logarithmically.

UCB's core idea is "Optimism in the face of uncertainty". Try the uncertain, maybe not promising enough action. 

Recall in greedy, we just pick the $a_t = \underset{a \in \mathcal{A}}{\text{argmax}} \; Q_t(a)$. We can add another $U_t(a)$ there that depends on the number of times $N_t(a)$ action $a$ has been selected.

If we pick that $U_{t}(a) = \sqrt{\frac{\log t}{2N_{t}(a)}}$, we can show by [[Hoeffding's Inequality]], that's quite good.

So the formula is:

$$
a_t=\underset{a\in\mathcal{A}}{\operatorname*{\operatorname*{argmax}}}Q_t(a)+c\sqrt{\frac{\log t}{N_t(a)}}
$$

## Thompson sampling

Really the reward we got from each action should be a distribution. UCB just ignores that though and uses count. 
Let's just model a distribution for the q function of each action.
Thompson sampling (Thompson 1933):
* Sample $Q_t(a) \sim p_t(q(a)), \forall a$
* Select action maximising sample, $A_t = \underset{a \in \mathcal{A}}{\text{argmax}} \; Q_t(a)$
* **Thompson sampling** is sample-based probability matching

$$
\begin{aligned}
\pi_t(a) &= \mathbb{E} \left[ \mathbb{I} (Q_t(a) = \max_{a'} Q_t(a')) \right] \\
&= p \left( q(a) = \max_{a'} q(a') \right)
\end{aligned}
$$

You can imagine for Bernoulli bandits, we can model it by Beta distribution. 

## Appendix: why UCB doesn't scale

The short answer: **UCB doesn't work "out of the box" in complex environments.**

The UCB algorithm's action selection is:

$$
a_t = \arg\max_{a} \left( Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \right)
$$

The most important part of that formula is **$N(a)$**, the "visit count" for that arm. In the MAB problem, this is easy. You only have one state, so you just count how many times you've pulled each arm.

Now, apply this to a full RL problem like an Atari game:

- **The State Space is Massive:** The "state" is the pixel data from the screen. The number of possible screen images is astronomical ($210 \times 160$ pixels, many colors).
- **You Never Visit the Same State Twice:** Because the state space is so large (effectively continuous), your agent will _never_ encounter the _exact same_ pixel-perfect state $s$ more than once.
- **The Count is Always 1 (or 0):** If you try to maintain a visit count $N(s, a)$ for every state-action pair, the count for _any_ state you are currently in will be 1. This makes the UCB formula's exploration bonus explode, and the agent just explores randomly.

So, while $\epsilon$-greedy is "dumb" exploration (it just picks a random action $\epsilon$ percent of the time), **it is simple and it scales.** It doesn't care how big the state space is.