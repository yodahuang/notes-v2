---
date: 2026-03-15
pdf: "[[cs336_lecture_02.mhtml]]"
---
## Numbers
- Training GPT-3 (2020) took 3.14e23 FLOPs
- Training GPT-4 (2023) is speculated to take 2e25 FLOPs
- A100 has a peak performance of 3.12e14 FLOP/s
- H100 is 9.90e14 FLOP/s

## Linear model

Forward: We have one multiplication (x[i][j] * w[j][k]) and one addition per $(i, j, k)$ triple.
Backward: 4 per $(i, j, k)$.

