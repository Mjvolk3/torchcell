---
id: p4qsahyb8bc52y22h73qn10
title: Test_neo4j_cell_process
desc: ''
updated: 1790716699423
created: 1790716699423
---

## 2026.09.29 - The process pipeline on a hand-built raw LMDB

Fixture: a three-record raw LMDB (fitness, gene interaction, metabolite) written under `tmp_path` with a four-gene `GeneSet`; `load_raw` is patched on `Neo4jCellDataset` to return that LMDB and to record its six arguments; the converter, deduplicator and aggregator are recording fakes that append a tag to `dataset_name`, so the finished `dataset_name_index` key `toy+conversion+deduplication+aggregation` proves the stage order and that the processed copy comes from the last stage. Expected values: the exact `load_raw` and stage-constructor argument tuples, the four `STAGE_COMPLETE` markers, the exact `label_df` rows (NaN for the dict-valued metabolite label), the three verbatim skip messages, the `Neo4jQueryRaw` kwargs (`io_workers=10`, `num_workers=10`, sorted gene set) and its three-line banner, `Neo4jCellDataset(2)`, and the local connection defaults.

Findings pinned (Phase 10 of [[test-campaign.2026.09.25]]): `overwrite_intermediates=True` removes a directory with `os.remove` and can never succeed; the worker-dropped cache tuple names a cache that is never filled; the index loops keep the first item of a partly malformed entry; the deletion-gene index has no error handler; a caller-supplied `"base"` graph replaces the gene-set nodes. Coverage of `torchcell/data/neo4j_cell.py`: 53.2% to 60.1%; the remaining 940 lines are the `main*` demo functions.
