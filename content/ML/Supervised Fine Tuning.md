---
aliases:
  - SFT
date: 2025-09-14
---
SFT is imitation learning, or in the very simple form, supervised learning.

> A common question we get is: isn’t this the same as pre-training, then? It is indeed very similar, with only one crucial difference: you only _condition_ on the prompt; you don’t _learn_ the prompt. Why? Because what you want is to **learn how to answer that question, not learn the question itself**.

> We do it by feeding the whole sequence through (including system prompt and user prompts as well as any special characters) without any masking, but when we compute the loss, we **mask** the loss of every token that is not the actual response.

> That said, the scale for these is nowhere near pre-training – you should expect to do SFT within a few million samples at the maximum, so only a few B tokens, whereas pre-training will crunch **trillions** of tokens.

Since it's supervised learning, who writes the response? Other than human, a common approach is to use the LLM to generate the response it would train on (note that's a upper bound). An approach from [[Llama 2]] is to do [[Rejection Sampling]]. We generate 10 answers from different prompts, seeds, etc. and pick the best answer and add that to the bank. 

> you can think of this approach as a way to kinda, sorta, distill from a kinda, sorta, ensemble into a single model (very handwavy, I know!).
