---
date: 2026-03-29
original title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
year: 2024
---
# MLA — Multi-head Latent Attention

Disclaimer: I haven't read the full paper.

![[mla.png]]

- Key idea: project down each key and value vector from $N*H$ dimensions to $C$ dimensions
- DeepSeek v2: reduce $N*H = 16384$ to $C = 512$
- Wrinkle: MLA is not compatible with [[Rotary position embedding|RoPE]], so need to add additional 64 dimensions for RoPE, so $512 + 64 = 576$ total dimensions

The rest of the note is based on my conversation with Claude Sonnet 4.6:

---

## Core Idea: Bottleneck + Cache Shift

Standard MHA caches K and V at full dimension. MLA instead:

1. **Compress** `x` → latent `c_KV` (small, cache this)
2. **Decompress** `c_KV` → K, V at inference time

> [!tip] Key insight
> We shift _where_ we cache. Instead of caching `K, V ∈ ℝ^d`, we cache `c_KV ∈ ℝ^{d_c}` where $d_c \ll d$. The decompression back to full rank happens on-the-fly and is **not** stored.

![[mla_bottleneck_cache.svg|637]]
## Dimensions

Let $d$ = `d_model`, $h$ = num heads, $d_h = d/h$ per-head dim.

| |KV cache / token|Expressiveness|
|---|---|---|
|Standard MHA|$2d$|Full rank|
|Shrink K/V heads|$2d_c$|Reduced (small heads)|
|**MLA**|$d_c$|**Full rank** (decompressed)|

**Compression:**

$$
c_{KV} = x W_{DKV}, \quad W_{DKV} \in \mathbb{R}^{d \times d_c}
$$

**Decompression:**

$$
K = c_{KV} W_{UK}, \quad V = c_{KV} W_{UV}, \quad W_{UK}, W_{UV} \in \mathbb{R}^{d_c \times d}
$$

## Is it equivalent to just shrinking K/V?

**No.** Shrinking K/V heads means each head genuinely has fewer dimensions — less expressive. MLA caches small but reconstructs full-rank K/V via learned up-projections. The bottleneck is in the _cache_, not in the attention computation.

## The Absorption Trick (avoid materializing K/V)

Naive MLA does 2 BMMs to get K and V before attention — worse than standard MHA. The trick: **absorb the up-projection weights into Q and output projection**.

$$
\text{scores} = Q W_{UK}^\top C_{KV}^\top = \underbrace{(Q W_{UK}^\top)}_{Q'} C_{KV}^\top
$$

So redefine $Q' = Q W_{UK}^\top$ (merged into $W_Q$, done once), then attention runs directly against cached latents $C_{KV}$:

$$
\text{out} = \text{softmax}(Q' C_{KV}^\top) \cdot C_{KV} W_{UV}^\top
$$

where $W_{UV}^\top$ is absorbed into the output projection $W_O$.

> [!note] Result 
> At inference, still **1 BMM against the cache** — same as standard MHA, but the cache is $d_c$ wide instead of $2d$. Pure win on memory, no extra compute.

## RoPE Caveat

RoPE is position-dependent, so it **cannot** be absorbed into a static weight matrix. MLA keeps a small separate $K_\text{rope}$ (a few dims per head) that is materialized explicitly and cached separately. This bypasses the compression for positional info only.