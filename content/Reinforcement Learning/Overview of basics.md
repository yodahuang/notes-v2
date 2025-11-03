---
date: 2025-10-04
---

I took deep RL course in grad school and this is a refreshment based on materials of [Hugging Face Deep Reinforcement Learning](https://huggingface.co/learn/deep-rl-course/unit0/introduction?fw=pt). Honestly it's not a good course. It makes hard parts feel hand-wavy and turns simple things overly complicated. This note serves as a place where I store my nodes, which should be a better starting point for exploring RL ideas. 

Edit: the course is just **useless**. It teach you the obvious, but hides the details from you, and the homework / projects are just using existing APIs. I strongly suggest DeepMind x UCL 2021 lecture series [here](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm). 

RL problem is optimization problem. So it shares quite some core idea with control theory. You should feel at home with state $S_t$ and action $A_t$, they are the same one in [[Intro and Bayes Filters|Bayes Filter]] or [[Kalman Filter]]. We got new thing $R_t$, which is used for reward / loss. 

The goal of the optimization is to maximize the expected cumulative reward. 

Well, here I think there are two things worth noticing. 
- It's an MDP—meaning my optimal policy depends only on my current observation. In other words, the state space should be designed cleverly so that it captures all necessary information.
- The problem has optimal substructure, meaning that if you divide the problem into smaller subproblems and solve each of them optimally, the overall solution will also be optimal. Well, that directly leads to Bellman's equation, and Bellman is the one who introduced dynamic programming. 

## Value-based methods

We can assign a value to each state, or to a state-action pair. The value captures how good the state is. Recall the markov property, so the value only needs to depend on state or stake action pairs, and nothing else really matters.  Let's call it $V$.

$$V_\pi(s)=\mathbf{E}_\pi[R_{t+1}+\gamma*V_\pi(S_{t+1})|S_t=s]$$

I shall comment that this is very obvious. The only less obvious part is the $\gamma$. 

If you're clever enough, you can think about deriving it from the end to the beginning. But wait—where do you know where to end? So why don't we just go in whatever order and do it multiple times until nothing changes? And there you get *Value Iteration*. (Imagine the value to be correct closest to the ending condition, and it "brushfires" back in iterations) 

But then we'll need the full state transition matrix. What if we don't know the state transition matrix? Can we just sample it? And then you get *TD(0)* learning. Time difference. 

$$V(S_t)\leftarrow V(S_t)+\alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]$$

We're just sampling actions. Can we do better with a [[EM]] idea? First we get value estimation, then we use the value estimation to get a better policy. That's *Q learning*. The next action is using epsilon greedy.

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma max_aQ(S_{t+1},a)-Q(S_t,A_t)]$$

See that $\max$ there? We are basically estimating the next state value by a greedy approach. Now Q learning is an off-policy method. If it uses the same policy to collect data, then that's SARSA.

These methods normally only work well in discrete space (for the $\max$ to work).

Another super simple way is *Monto Carlo Approach*. You get the entire episode of experience, instead of iterative, and use the final reward to update your belief.

### Online / offline / On-policy / Off-policy

On-policy vs Off-policy → 🔸 *What policy are you learning about vs what policy generated the data?*

| Term           | Meaning                                                                              |
| -------------- | ------------------------------------------------------------------------------------ |
| **On-policy**  | You learn the value of **the same policy** you use to collect data.                  |
| **Off-policy** | You learn the value of **a different (target) policy** than the one collecting data. |
Online vs Offline RL → 🔸 *When / how is the data collected and used for learning?*

| Term          | Meaning                                                                                                    |
|---------------|-------------------------------------------------------------------------------------------------------------|
| **Online RL** | You **interact with the environment as you learn** — collect experience and update your policy in real time. |
| **Offline RL**| You’re given a **fixed dataset** of interactions beforehand — no further environment interaction allowed.   |

|                | **On-policy**                         | **Off-policy**                                |
| -------------- | ------------------------------------- | --------------------------------------------- |
| **Online RL**  | SARSA, Policy Gradient methods        | Q-learning, DQN (ε-greedy behavior)           |
| **Offline RL** | Behavior cloning, fitted policy eval. | Offline Q-learning, batch-constrained methods |
