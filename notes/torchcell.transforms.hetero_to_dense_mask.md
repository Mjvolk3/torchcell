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

## 2026.09.30 - Explicit square_attrs replaces the shape guess (issue #570)

- Previous behavior: any node tensor with `size(0) == N` and `size(1) == N` was padded on both axes, so a feature dimension that happens to equal the node count was padded as if it were a node axis.
- Fix: the constructor takes `square_attrs: dict[node_type, list[str]]` (default empty). Only listed attributes are padded on both axes; every other node tensor with `size(0) == N` is padded on rows only. A listed attribute that is absent or not `[N, N, ...]` raises `ValueError` naming the node type, the attribute, the expected shape and the actual shape. `__repr__` shows `square_attrs` when set.
- Live consumers: `torchcell/models/hetero_cell_nsa.py` (~481, ~499), `torchcell/models/hetero_cell_nsa_retry.py` and `torchcell/data/neo4j_cell.py` (~1482) pass only `num_nodes_dict`. By code reading, their gene node stores carry `x`, `x_pert`, `pert_mask`, `mask`, `node_ids`, `perturbation_indices`, `ids_pert`, none of them `[N, N]`, so they need no `square_attrs` and their output is unchanged (not measured on a live sample).
- Evidence: `tests/torchcell/transforms/test_hetero_to_dense_mask.py` (`test_an_unlisted_feature_dimension_equal_to_n_is_padded_on_rows_only`, `test_padding_extends_pos_and_node_sized_tensors_and_leaves_the_rest`, `test_square_attrs_apply_only_to_their_node_type`, `test_a_listed_attribute_that_is_not_square_is_refused_by_name`).
- Review follow-up: a reserved name (`x`, `pos`, `mask`, `num_nodes`, `node_ids`, a leading underscore; `RESERVED_NODE_ATTRS`) in `square_attrs` is refused at construction, since `x`/`pos` are padded on rows only and the rest are skipped, so listing one would silently not be squared. A `square_attrs` node type absent from the data raises in `forward` naming it and the data's node types. A missing listed attribute reports `is missing from the <node type> store`. Call sites checked by code reading, all passing only `num_nodes_dict` and carrying no `[N, N]` node tensor: `torchcell/models/hetero_cell_nsa.py` (2), `torchcell/models/hetero_cell_nsa_retry.py` (2), `torchcell/data/neo4j_cell.py`, `experiments/003-fit-int/scripts/hetero_cell_nsa.py:387`, `experiments/006-kuzmin-tmi/scripts/hetero_cell_nsa_retry.py:299`, `torchcell/scratch/load_batch.py`, `load_batch_004.py`, `load_batch_005.py`.
