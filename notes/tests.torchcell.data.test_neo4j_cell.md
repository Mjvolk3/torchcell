---
id: amtblyv6uphh2zdthe1uydv
title: Test_neo4j_cell
desc: ''
updated: 1790533870719
created: 1790533870719
---

## 2026.09.27 - The pure helpers and the pickling contract

The six pickling tests (Phase 2: `__getstate__` drops the LMDB environment and every bulk cache, keeps the rest, does not mutate the live dataset, and the pickle is small) plus fourteen exact-value tests on the module's helpers: `min_max_normalize_embedding` on a 2x3 tensor with a constant column gives `[[0, 0.5, 0], [1, 0.5, 1]]`; `normalize_tensor_row`; `min_max_normalize_dataset` on a duck-typed `Data`-backed fake for 2-D and flat storage; `create_embedding_graph` (exact node embeddings, a gene outside the set dropped); `create_graph_from_gene_set`; `parse_genome`; `ParsedGenome`; the `ProcessingStep` values; `_determine_processing_steps` for all eight converter, deduplicator and aggregator combinations; `_get_lmdb_path` for all five steps; the `gene_set` getter and setter error paths. Findings: the validator message "gene_set must be a GeneSet" is unreachable because pydantic's arbitrary-type check raises "Input should be an instance of GeneSet" first (neo4j_cell.py lines 61 to 65); `normalize_tensor_row` does not make rows sum to one, its `clamp(min=1.0)` leaves rows whose range-sum is below one unscaled, it returns a list of rows, and a list input keeps only its first tensor (lines 73 to 82); `min_max_normalize_dataset` normalizes only the first embedding key while its docstring says the entire dataset (line 114). The item path on a hand-written processed store is in [[tests.torchcell.data.test_neo4j_cell_hermetic_build]]. Phase 6 of [[plan.test-suite-buildout.2026.09.25]].
