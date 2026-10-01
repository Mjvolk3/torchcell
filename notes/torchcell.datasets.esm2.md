---
id: kycvniljkk7kiyze4852ie2
title: Esm2
desc: ''
updated: 1790813170649
created: 1790813170649
---

## 2026.09.30 - Leading axis, exclusion case, unknown names (issue #543)

- **Layout.** Each gene used to be squeezed to `[D]`, so the collate concatenated all genes into one flat `[n_genes * D]` vector and `ds["<gene>"]` (which slices `[i:i+1]`) returned one scalar of the wrong gene. Each gene now stores `[1, D]` (`reshape(1, -1)`), the collate is `[n_genes, D]`, and integer and gene-id lookup return the same `[1, D]` row. `CellDataset.create_embedding_graph` (`squeeze(0)` per row) still yields the gene's `[D]` node feature, and `neo4j_cell.min_max_normalize_dataset` already handles both layouts, so existing flat stores keep working.
- **Exclusion.** The `_no_dubious_uncharacterized` variants listed `dubious` / `uncharacterized` in lowercase and never matched SGD's `Dubious` / `Uncharacterized`; they are now capitalized.
- **Names.** The exclusion list was read from `MODEL_TO_WINDOW` before `BaseEmbeddingDataset` validated the name, so an unknown name was a bare `KeyError` and `model_name=None` could not construct. The lookup moved into `process`, after its `None` return, so an unknown name is the base `ValueError` naming the valid names, and `None` builds neither a backbone nor a store.
- **Backbone.** Built only inside `process` (PyG runs it only when the store is absent); the dead post-`super` rebuild branch was removed.

A rebuilt store differs on disk: `[n_genes, D]` instead of flat, and the `_no_dubious_uncharacterized` stores zero Dubious and Uncharacterized genes. Tests: [[tests.torchcell.datasets.test_esm2]].
