---
id: 548ch6gj480a8p8e1swrzy6
title: Hetero_to_dense_mask
desc: ''
updated: 1790816109413
created: 1790816109413
---

## 2026.09.30 - Square per-node padding and the original-count edge filter (issue #538)

- A per-node `[N, N]` tensor was padded on rows only and came back `[N_pad, N]`; a tensor whose second axis also equals the original node count is now padded on both axes. A feature axis that happens to equal `N` is padded too, since the transform cannot tell it from a node axis.
- The validity filter compared indices with the PADDED node count, so an edge into a padding row set `adj_mask` on a row `mask` marks as padding. Both `adj_mask` and `inc_mask` now keep only entries whose endpoints are below the ORIGINAL node counts; the sparse indices are untouched.
- Evidence: `tests/torchcell/transforms/test_hetero_to_dense_mask.py` (`test_padding_extends_pos_and_node_sized_tensors_and_leaves_the_rest`, `test_an_edge_touching_the_padding_row_is_dropped_from_the_mask`).
