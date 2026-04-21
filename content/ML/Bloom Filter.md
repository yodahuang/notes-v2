---
date: 2026-04-07
---
This note starts with me dictating with Claude Sonnet 4.6 doing touch ups and the actual math.

---

# Bloom Filter

We want a Hash Set, but that's too large. If we can tolerate a little false positive rate, we can have a very memory-efficient approximate membership structure: a **Bloom Filter**.

## Core Idea

A Bloom Filter starts with a common trick: hash a value and perform a mod operation to get a small index. You then have a **bit array** of size $m$ and simply set that bit to 1.

Then you do some math and figure: "Hey, that actually has a pretty high error rate." A single hash function means a single collision causes a false positive.

So, why not use multiple hash functions? When testing whether a value belongs to the set, ask: **for all $k$ hash functions, are all $k$ bits set to 1?** A false positive now requires all $k$ positions to be coincidentally occupied — much rarer.

> [!tip] Key insight
> You don't even need $k$ separate bit arrays. You can combine them into **one big bit array** of size $m$. Each hash function maps into the same array. It can be shown mathematically that $k$ small arrays of size $m/k$ versus one big array of size $m$ are equivalent in false positive rate.

## Operations

**Insert** a key $x$:

$$
 \text{for } i = 1, \ldots, k: \quad B[h_i(x) \bmod m] \leftarrow 1 
$$

**Query** for key $x$:

$$
 \text{return } \bigwedge_{i=1}^{k} B[h_i(x) \bmod m] = 1 
$$

> [!note]
> 
> - **No false negatives**: if $x$ was inserted, all its bits are set, so it always returns true.
> - **False positives possible**: another set of keys may have coincidentally set all $k$ positions.
> - **No deletion**: clearing a bit might unset a bit shared by another key.

## False Positive Rate

**Setup:** $m$ bits, $k$ hash functions, $n$ elements inserted.

**Step 1 — Probability a specific bit is still 0 after inserting one element:**

Each hash function sets one bit uniformly at random. The probability a given bit is _not_ set by one hash function on one insertion is:

$$
 1 - \frac{1}{m} 
$$

After $k$ hash functions and $n$ insertions ($kn$ total bit-set operations):

$$
 P(\text{bit} = 0) = \left(1 - \frac{1}{m}\right)^{kn} 
$$

**Step 2 — Apply the standard limit** $\left(1 - \frac{1}{m}\right)^m \approx e^{-1}$ for large $m$:

$$
 P(\text{bit} = 0) \approx e^{-kn/m} 
$$

So the probability a bit is 1 (occupied by some prior insertion) is:

$$
 P(\text{bit} = 1) \approx 1 - e^{-kn/m} 
$$

**Step 3 — False positive rate.** A false positive occurs when all $k$ probed bits happen to be 1:

$$
 \boxed{\varepsilon = \left(1 - e^{-kn/m}\right)^k} 
$$

## Optimal Number of Hash Functions

For fixed $m$ and $n$, minimize $\varepsilon$ over $k$. Taking $\frac{d\varepsilon}{dk} = 0$ gives:

$$
 k^* = \frac{m}{n} \ln 2 
$$

Substituting back, the false positive rate at optimal $k$ simplifies to:

$$
 \varepsilon^* = \left(\frac{1}{2}\right)^{k^*} = \left(\frac{1}{2}\right)^{(m/n)\ln 2} 
$$

Or equivalently, to achieve a target false positive rate $\varepsilon$, the required bits per element is:

$$
 \frac{m}{n} = -\frac{\log_2 \varepsilon}{\ln 2} \approx -1.44 \log_2 \varepsilon 
$$

> [!example] Rule of thumb
> For $\varepsilon = 1%$: $m/n \approx 9.6$ bits per element, $k^* \approx 7$ hash functions. For $\varepsilon = 0.1%$: $m/n \approx 14.4$ bits per element, $k^* \approx 10$ hash functions.