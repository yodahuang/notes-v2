---
date: 2025-11-02
aliases:
  - Off-Policy
  - On-Policy
---
### **On-policy** learning
- Learn about **behaviour** policy $\pi$ from experience sampled from $\pi$

### **Off-policy** learning
- Learn about **target** policy $\pi$ from experience sampled from $\mu$
- Learn ‘counterfactually’ about other things you could do: “what if...?”
  - E.g., “What if I would turn left?” $\implies$ new observations, rewards?
  - E.g., “What if I would play more defensively?” $\implies$ different win probability?

Evaluate target policy $\pi(a \mid s)$ to compute $v_\pi(s)$ or $q_\pi(s, a)$  
While using behaviour policy $\mu(a \mid s)$ to generate actions  
#### Why is this important?
- Learn from observing humans or other agents (e.g., from logged data)  
- Re-use experience from old policies (e.g., from your own past experience)  
- Learn about **multiple** policies while following **one** policy  
- Learn about **greedy** policy while following **exploratory** policy  
