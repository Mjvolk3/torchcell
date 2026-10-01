---
id: jb7r06xqj8ndtua4lkksp5b
title: Lazy_collate
desc: ''
updated: 1790872868966
created: 1790872868966
---

## 2026.10.01 - PyG batch bookkeeping and follow_batch (issue #572)

Previous behavior: `lazy_collate_hetero` wrote only `batch` per node type, ignored `follow_batch`, and returned a one-sample list unbatched. This went unnoticed because until PRs #549 and #571 the datamodules handed `collate_fn` to PyG's `DataLoader`, which discards it and collates with its own `Collater` (with the datamodule's `follow_batch`). Once the datamodules honored the custom collate, the 006 lazy model received batches with no `gene.ptr` (read by `forward_single` as `len(ptr) - 1`, an `AttributeError`) and no `perturbation_indices_ptr` / `perturbation_indices_batch` (so the local predictor ran with `batch_assign=None`).

Fix: every node type gets `batch` and `ptr`; every `follow_batch` key a node type carries gets `<key>_batch` and `<key>_ptr` over its rows; the batch carries `num_graphs`; a one-sample list is collated the same way. `LazyCollater` passes its own `follow_batch` to it. A key no node type carries is skipped, as PyG does. Tests in [[tests.torchcell.datamodules.test_lazy_collate]] assert the exact vectors and that they equal `Batch.from_data_list(..., follow_batch=...)` on the same samples.
