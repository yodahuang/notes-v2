---
date: 2026-04-09
---
$$
Pr[h(A) = h(B)] = Jaccard(A, B)
$$
How we do this? Hash all the items in a set and pick the min (or max, doesn't matter). See [[Jaccard similarity]] for what's that Jaccard.

```python
def minhash(S: set[str], seed: int):
	return min(mmh3.hash(x, seed) for x in S)
```

Why? Sorted hash function over a heap is basically random permutation, or shuffle.

Say $A$ and $B$ set both have 5 elements, 3 of them are shared. $A: \{\alpha_{1}, \alpha_{2}, \alpha_{3}, \beta, \gamma\}, B: \{\alpha_{1}, \alpha_{2}, \alpha_{3}, \omega, \delta\}$,  If one of the three $\alpha$ produce the smallest hash among the 7 unique elements, then hash collides, $Pr = 3/7$.
