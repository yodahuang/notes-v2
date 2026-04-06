---
date: 2026-03-29
pdf: "[[cs336_lecture_10.mhtml]]"
---
[[KV Cache]] 

### Latency & throughput

Latency is determined by memory IO (read all parameters and KV cache for each step).
```python
latency = memory / memory_bandwidth
```
Throughput is the inverse of latency, but we're generating B tokens in parallel.
```python
throughput = B / latency
```

**Tradeoff** between latency and throughput:
1. Smaller batch sizes yields better latency but worse throughput
2. Larger batch sizes yields better throughput but worse latency

Easy parallelism: if you launch M copies of the model, latency is the same, throughput increases by M!
Harder parallelism: shard the model and the KV cache. See [Scaling book chapter on Transformers](https://jax-ml.github.io/scaling-book/inference/)

