---
id: xaoeg9l1kfdmt9q1r5b7nu6
title: Bacteria_released_columns_store_report
desc: ''
updated: 1791863772400
created: 1791863772400
---

## 2026.10.12 - Counts and L0-L4 of the PR #1072 stores

Reads the 13 stores slurm array 4475 rebuilt for #879, #904, #905, #914 and #915: each store's LMDB entry count and its `preprocess/verification_report.json`, against the counts measured before the rebuild. Writes `experiments/036-dataset-fixes-before-kg-build/results/bacteria_released_columns_store_report.json` and prints one L0-L4 table per store, the tables the dataset notes carry. `--root CLASS=PATH` reads a superseded sibling when another branch has since rebuilt the shared dev store (Rousset 2018, rebuilt by #1082 after this array). Result: no store decreased; every gate rule passes on all 13.
