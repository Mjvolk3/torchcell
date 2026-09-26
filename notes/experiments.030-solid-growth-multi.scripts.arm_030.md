---
id: r5gm5ldtb7yhvycw58o0682
title: Arm_030
desc: ''
updated: 1790417000227
created: 1790417000227
---

## 2026.09.26 - The arm definition every 030 per-entry script shares

`arm_030.py` is imported by the training script, the split-cache warmer, the normalization-stats writer and the smoke report so that "the records that trained" and "the records the constants were fitted on" are one computation. `resolve_arm(subset_cfg)` reads the committed artifacts (`subset_S3_indices.json.gz`, `pinned_splits_from_010_seed_42.json.gz`, `essentiality_holdout_030.json.gz`, `subset_definitions_030_summary.json`) and returns an `Arm030` (pydantic): the pool (S3 minus the 698 holdout records, 1,120,964), the pinned split, the holdout sets with labels by gene, and the sorted 15-name token vocabulary (plus `SyntheticOffsetDataset` under the smoke). `train_records()` is the exact training set under `unpinned_to_train` (1,045,618 records) and `index_sha256` its fingerprint. `build_dataset` constructs the genome, graph, embeddings and `Neo4jCellDataset` with `Perturbation(dataset_vocabulary=...)` and no deduplicator (the 030 build merged nothing); `make_data_module` builds the `CellDataModule` with the holdout as `extra_val_indices={"val_ess": ...}` and asserts the realized splits equal the arm's. `run_smoke_check` measures a trained model under both tokens (`SmokeResult`), read by [[experiments.030-solid-growth-multi.scripts.smoke_report_030]].

Design decisions: the vocabulary is derived from the artifact and `vocab_size` is never typed into a config; the holdout genes are mapped to node indices through `cell_graph["gene"].node_ids` at run time and a missing gene raises.
