---
aliases:
  - PPO
Arxiv: https://arxiv.org/abs/1707.06347
pdf:
original title: Proximal Policy Optimization Algorithms
date: 2025-09-14
tags:
year: 2017
---
Note that I haven't finished the paper (it would require prior review of [[TRPO]] etc.). A good place to start may be [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) and [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html#). RL is so back. 

The rest of the note is based on Appendix A: Diving deeper into PPO of this [blog post](https://pytorch.org/blog/a-primer-on-llm-post-training/).

We start with [[REINFORCE]], then [[Actor-Critic]], then [[TRPO]] and finally reach PPO.

Without consulting the actual math that leads to REINFORCE (that aims to maximize the expected episodic reward), we basically want this if the `model` is the policy network.
```python
model.weights.grad[good_actions] += delta
model.weights.grad[bad_actions] -= delta
```
so we can see the "cost function" is simply `log_probs * per_token_reward`

The overall code look like this (notice this is very basic REINFORCE)
```python
for batch in dataloader:  # Iterate over dataset (prompts)
    prompts = batch        # Get input prompts: (bsz, prompt_lens). Can be ragged, or packed.
    responses, log_probs = model.generate(prompts)  # Autoregressive generation! MANY forward calls, need KV cache etc.).

# Also note: we NEED to return log_probs for all the intermediate generations and note that we are not detaching them from the graph as we are gonna need them later.

   # responses and log_probs are of size (bsz, response_lens). Also ragged/packed. 

    sequence_rewards = get_feedback(responses)  # Get rewards (e.g., human
preference or heuristic).

    # Note that rewards CAN be negative, in that case this sign will be negative.

    # This is a tensor of size (bsz,). 

    per_token_reward = discount(sequence_rewards)  # This one is (bsz, response_lens). A simple way to discount is to multiply tokens by a factor gamma (eg 0.99).  So the last token gets reward of 1, the second-to-last gets 1*gamma, then the previous gets 1*gamma*gamma and so on. You don't want to maximize only your reward at time t but the sum of rewards till the end of the episode 

    optimizer.zero_grad()  # Reset gradients

    # Manually nudge log-probs based on reward signal

    adjusted_log_probs = log_probs * per_token_reward # this is a stochastic estimator for the gradient of the reward expectation given your stochastic policy - in other words: on average, the gradient of that thing points to where the policy is doing good!

    loss = -adjusted_log_probs.sum()  # Equivalent to maximizing probability of good actions

    loss.backward()  # Still ONE backward call! PyTorch knows what to do.

    optimizer.step()  # Update model parameters
```

And then there's PPO stuff, including first using REINFORCE with baseline, introducing value network for advantage calculation. Then standardize the advantages, introducing trust region (comparing with previous policy) and clip the large update, and finally taking a min. I believe it need to be understood more in depth with hands on training.

$$
\begin{aligned}
L_{\mathrm{PPO}}&=-\mathbb{E}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{\mathrm{old}}(a|s)}A,\mathrm{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\mathrm{old}}(a|s)},1-\epsilon,1+\epsilon\right)A\right)\right] \\
L_{\mathrm{value}}&=\mathbb{E}\left[\left(V_\theta(s)-G_t\right)^2\right]\\
L_{\mathrm{KL}}&=\mathbb{E}\left[D_{\mathrm{KL}}\left(\pi_{\mathrm{reference}}(a|s)||\pi_\theta(a|s)\right)\right] \\
L_\mathrm{total}&=L_\mathrm{PPO}+c_1L_\mathrm{value}-c_2L_\mathrm{KL}
\end{aligned}
$$