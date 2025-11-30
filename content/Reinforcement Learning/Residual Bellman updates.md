This is related to the TD equation in [[Function approximation]]. Note how we do not derive the formula for TD for $\delta w_t$. We just say "replace this part in MC with TD estimation". That's because we are not really minimizing the real loss: $\mathbb{E}[\delta_{t}^2]$. 

Now recall

$$
\delta_{t}= R_{t+1} + \gamma v_w(s_{t+1}) - v_w(S_t)
$$

We can compute out the "more sound" update:

$$
\Delta \mathbf{w}_t = \alpha \delta_t \nabla_{\mathbf{w}} \left( v_{\mathbf{w}}(S_t) - \gamma v_{\mathbf{w}}(S_{t+1}) \right)
$$

This tends to work **worse** in practice. It smooth both state (from and to).

We can also minimize the Bellman error directly (L1 loss).

**loss:**  

$$\mathbb{E}[\delta_t]^2$$

**update:**  

$$\Delta w_t = \alpha \delta_t \nabla_w (v_w(S_t) - \gamma v_w(S'_{t+1}))$$
 
...but requires a second independent sample $S'_{t+1}$ which could (randomly) differ from $S_{t+1}$ . (So we can’t use this online)