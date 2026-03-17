---
date: 2026-03-15
---

```python
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: DeviceLikeType
) -> tuple[Tensor, Tensor]:
    # Why are we writing this by hand instead of using DatasetLoader?
    # Note we are using "sampling", instead of multiple workers pre-fetching.
    # Called repeatedly each training step instead of using a stateful dataloader.
    # Works because:
    # (1) the full token array fits in RAM so sampling is just a random
    #   index + slice — no I/O to hide with prefetch workers;
    # (2) LLM training never "finishes" an epoch, so stateless random sampling is simpler and equivalent.
    # Note how this is different from vision tasks, where each input samples may be large (while here it's just text)

    dataset_token_count = len(dataset)
    # Exclusive upper bound since we need to shift by one for targets
    start_indices = np.random.randint(0, dataset_token_count - context_length, size=batch_size)
    # we want start_indices:start_indices+context_length
    # Construct one manually with advanced indexing
    # Broadcasting!
    indexing_arr = einx.add(
        "batch_size, context_length -> batch_size context_length", start_indices, np.arange(context_length)
    )
    # We can put them in pinned memory and do async copy.
    # But then we add memory pressure and slower allocation.
    # There exist stream sync but we are not overlapping data loading and training anyway.
    return (torch.from_numpy(dataset[indexing_arr]).to(device), torch.from_numpy(dataset[indexing_arr + 1]).to(device))

```

```python
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # Not that trivial

    # We can't clip for each p in parameter. That'll change the gradient direction.
    # We can't just clip the param themselves, since the max is specified on l2, not l1
    # So we need to work backwards: how can I scale my norm with a variable so it stays below?
    
    # We could do sum of torch.square ourselves to do param-norm,
    # or we could use a nice property of L2 norm (not for L3 and others),
    # norm of subnorm is still that norm (that's because sqrt cancels ^2).
    # Similar performance
    param_norm = torch.linalg.vector_norm(
        torch.tensor([torch.linalg.vector_norm(p.grad) for p in parameters if p.grad is not None])
    )

    # We can get this by solving ||g x|| = M
    gradient_scale_factor = max_l2_norm / (param_norm + 1e-6)

    for p in parameters:
        if p.grad is None:
            continue
        # We only need to clip, bu not pull small gradients up
        # Avoids one if.
        gradient_scale_factor = torch.clip(gradient_scale_factor, max=1.0)
        p.grad *= gradient_scale_factor
```