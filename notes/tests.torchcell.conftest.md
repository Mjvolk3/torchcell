---
id: 1zeoa12gcyxbb69sjhvea4u
title: Conftest
desc: ''
updated: 1790409479197
created: 1790409479197
---

## 2026.09.26 - Shared synthetic fixtures

`tests/torchcell/conftest.py` holds the in-memory fixtures the model and trainer tests share so no test needs `DATA_ROOT`: the Cell Graph Transformer `cell_graph` (8 genes, 4 reactions, 3 metabolites; gene-gene, gpr and rmr edges) and `batch` (3 genotypes perturbing {1,2}, {3}, {0,4,5}), lifted from `test_equivariant_cell_graph_transformer.py`; `fake_txn`, a dict-backed stand-in for the interned LMDB write transaction; and `dcell_graph` / `dcell_batch`, a root GO term with two stratum-1 children over four genes in the `go_gene_strata_state` layout `torchcell.models.dcell.DCell` reads, with the batch's `state` column flipped to 0 for knocked-out genes. Test modules cannot import a conftest (no `__init__.py` in the test tree, so a relative import has no parent package); they take the graph from the fixtures and restate the sizes they assert on.

## 2026.09.30 - Build-time sha256 pin fixtures

Two fixtures for the loader pin check (issues #518, #524, #528, #537). `raw_pin_calls` (module scope, autouse) replaces `verify_raw_files` in the 30 pinned loader modules with a recorder that demands each pinned file exist in `raw/` and records `(module, raw dir, pins)`, because the synthetic raw files the loader and adapter tests build from cannot carry the real pins. `off_pin_raw` restores the real check on one loader module and fills a fresh root's `raw/` with off-pin bytes, returning the root and their sha256, so each loader's refusal test drives the constructor path PyG takes when `raw/` is already populated.
