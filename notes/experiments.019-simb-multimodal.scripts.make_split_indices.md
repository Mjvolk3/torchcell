---
id: jx4vaypfchwgvy66oxr1apc
title: Make_split_indices
desc: ''
updated: 1789288479668
created: 1789288479668
---

## 2026.09.13 - Partitions materialized and hashed for the v13 split round

Builds the `CellDataModule` partition for each seed exactly as `train_cgt_multitask.py` does (same dataset root, query, deduplicator, aggregator, `split_indices` keys) and records, per seed, record counts, labelled expression counts and the sha256 of `index_seed_<k>.json` in `results/split_indices_manifest.json`. Run for seeds 0-3:

| seed | records train/val/test | expression train/val/test | index sha256 | status |
|---|---|---|---|---|
| 0 | 4,074 / 534 / 517 | 1,244 / 155 / 155 | `a572f3eb10f5` | existed, unchanged |
| 1 | 4,059 / 529 / 537 | 1,253 / 151 / 150 | `0100f1fab2f0` | existed, unchanged |
| 2 | 4,069 / 529 / 527 | 1,244 / 155 / 155 | `e6d4d87fe0ff` | existed, unchanged |
| 3 | 4,069 / 529 / 527 | 1,244 / 155 / 155 | `bced67c62b3a` | new |

The IGB copies of seeds 0-2 carry the same hashes; seed 3 was copied to IGB by `scp` before the round was submitted, so the GPU runs ([[experiments.019-simb-multimodal.conf.cgt_expr_v13_split]]) and the CPU baselines ([[experiments.019-simb-multimodal.scripts.expression_baselines_split]]) read one file per split. The split is random over records within each index key (80/10/10), not disjoint folds, so the four draws measure the spread of the absolute number, not a K-fold estimate.
