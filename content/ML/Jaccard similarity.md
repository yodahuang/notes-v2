---
date: 2026-04-09
---

$$
\text{Definition: } \text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

```python
def compute_jaccard(A, B):
    intersection = len(A & B)  # Elements in both sets: {"1", "2", "3"}
    union = len(A | B)         # All unique elements: {"1", "2", "3", "4", "5"}
    return intersection / union

# A and B must be set objects in Python
set_A = {"1", "2", "3", "4"}
set_B = {"1", "2", "3", "5"}

jaccard = compute_jaccard(set_A, set_B)
```
