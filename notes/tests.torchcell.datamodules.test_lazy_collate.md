---
id: sh7khfigr1pylx0j4f709o2
title: Test_lazy_collate
desc: ''
updated: 1790872930156
created: 1790872930156
---

## 2026.10.01 - Lazy collate bookkeeping (issue #572)

New file. Three lazy samples perturbing {1}, {2, 3}, and nothing: `ptr` [0, 4, 8, 12], `perturbation_indices_batch` [0, 1, 1], `perturbation_indices_ptr` [0, 1, 3, 3], `x_batch` / `x_ptr`, offset edge indices, `num_graphs` 3, all equal to `Batch.from_data_list` with the same `follow_batch`; an absent `x_pert` is skipped. Without `follow_batch` only `batch` and `ptr` are added. A one-sample list is collated, not returned as is.

## 2026.10.06 - Phase 21: collate branches and verify_batch_structure

Fixture: the 4-gene lazy sample plus a gene -> reaction hyperedge relation (2 reactions per sample), so sample i shifts gene indices by 4 i and reaction indices by 2 i: hyperedge_index [[0, 3, 4, 7], [0, 1, 2, 3]].

- Empty list refused; tensor edge attributes concatenated; non-tensor edge and node attributes kept as per-sample lists; a key that mixes a tensor and a string becomes the list [tensor([7]), 'gpr']; a key missing from one sample raises KeyError; a store with a mask but no index is dropped with its mask; non-lazy HeteroData and plain Data go to PyG's Collater.
- `verify_batch_structure` prints and returns False with the exact message for each check (batch max, batch size, source bound, dest bound at the boundary 12 >= 12, mask length); hyperedge_index is bounds-checked; an index-free store is skipped.

Findings:

- The docstring's check 1 ("properly offset, no overlap between graphs") is not implemented: a batch whose edges all point into graph 0 passes.
- A relation with zero edges raises RuntimeError from `.max()` on an empty tensor (lazy_collate.py:268), which `except AssertionError` does not catch.
- Every check is an `assert`, so under `python -O` the verifier returns True for a wrong batch (reproduced in a subprocess).

## 2026.10.06 - Phase 21 audit 2

Reach of the three verify_batch_structure findings: latent; the function has no caller in torchcell/ or experiments/.
