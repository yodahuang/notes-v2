---
aliases:
  - DPO
Arxiv: https://arxiv.org/abs/2305.18290
pdf:
original title: Direct Preference Optimization:Your Language Model is Secretly a Reward Model
date: 2025-09-14
tags:
year: 2023
---
The rest of the content is not based on the paper (I haven't finished it yet), just based on a [blog post](https://pytorch.org/blog/a-primer-on-llm-post-training/).  To understand what's really going on, we need to first understand the [[PPO]] paper.

DPO is currently specifically used for [[RLHF]], but really it's an offline preference-based RL algorithm. It somehow converts the long and unstable cycle of RL to a supervised learning problem. The only thing you need is pairwise preference, and it'll do a binary logistic regression-like loss (but not exactly), and tell you how to optimize the model. It reuses the model in training as the "critic" model in reward modeling.