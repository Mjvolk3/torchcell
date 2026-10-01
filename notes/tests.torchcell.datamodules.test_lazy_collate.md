---
id: sh7khfigr1pylx0j4f709o2
title: Test_lazy_collate
desc: ''
updated: 1790872930156
created: 1790872930156
---

## 2026.10.01 - Lazy collate bookkeeping (issue #572)

New file. Three lazy samples perturbing {1}, {2, 3}, and nothing: `ptr` [0, 4, 8, 12], `perturbation_indices_batch` [0, 1, 1], `perturbation_indices_ptr` [0, 1, 3, 3], `x_batch` / `x_ptr`, offset edge indices, `num_graphs` 3, all equal to `Batch.from_data_list` with the same `follow_batch`; an absent `x_pert` is skipped. Without `follow_batch` only `batch` and `ptr` are added. A one-sample list is collated, not returned as is.
