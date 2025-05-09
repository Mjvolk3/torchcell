---
id: kfzjq6vwcn4l67z5258uans
title: 152511 Batch_size
desc: ''
updated: 1746649934489
created: 1746649521010
---

breakpoint at

```python
interaction_scores = self.hyper_sagnn(
    perturbed_embeddings, batch["gene"].perturbation_indices_batch
)
```

`batch_size=2`

```python
perturbed_embeddings.size()
torch.Size([6, 64])
batch["gene"].perturbation_indices_batch.size()
torch.Size([6])
adjusted_indices.size()
torch.Size([6])
batch["gene"].perturbation_indices_batch
tensor([0, 0, 0, 1, 1, 1])
adjusted_indices
tensor([1312, 3149, 4147, 8719, 9197, 9670])
expanded_embeddings.size()
torch.Size([13214, 64])
```

`batch_size=4`

```python
perturbed_embeddings.size()
torch.Size([12, 64])
batch["gene"].perturbation_indices_batch.size()
torch.Size([12])
adjusted_indices.size()
torch.Size([12])
batch["gene"].perturbation_indices_batch
tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
adjusted_indices
tensor([ 1520,  3515,  6344,  8112,  8464, 10144, 14007, 15463, 19740, 21995,
        24529, 26293])
expanded_embeddings.size()
torch.Size([26428, 64])
```
