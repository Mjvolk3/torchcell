---
id: 5x7ps37z31lxfl1gscu6j76
title: Jackson2020_store_inventory
desc: ''
updated: 1791733104308
created: 1791733104308
---

## 2026.10.11 - Counts of the built Jackson 2020 store

`experiments/036-dataset-fixes-before-kg-build/scripts/jackson2020_store_inventory.py` streams the dev store `data/torchcell/tf_deletion_pseudobulk_jackson2020` (read only) and its `preprocess/release_ledger.json`, and writes `results/jackson2020_store_inventory.json`: records and cells per condition, reference cells per condition, the per-record spreads of cells, UMIs and nonzero genes, the marker UMIs, the empty (strain, condition) pairs, the Table S2 disagreements, the legacy feature ids and the raw-mirror pins. Every non-verifier count in [[torchcell.datasets.scerevisiae.jackson2020]] is read from that JSON. Run after a rebuild:

```bash
PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/jackson2020_store_inventory.py
```
