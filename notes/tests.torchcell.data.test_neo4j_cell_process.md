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

## 2026.09.30 - Phase 12: label_df on aggregated records

One test added (14 to 15): `label_df` keeps the last present value of a key that holds several experiments (fitness 0.8 then 0.5 gives 0.5), a later `environment_response` of None does not erase an earlier -1.25, and a categorical-only record stays NaN. Finding: `phenotype_label_index` lists that NaN record under `environment_response` because it reads only the stored `label_name` (`neo4j_cell.py` line 755), so a split drawn from the index can hold a record with no scalar target. The module stays at 60.1 percent: lines 1082 to 2021 are the `main*` demos (Neo4j, the genome, `experiments/003-fit-int/queries/*.cql`), line 64 is dead, `498 -> 501` cannot happen because the pipeline never schedules RAW as a next step.
