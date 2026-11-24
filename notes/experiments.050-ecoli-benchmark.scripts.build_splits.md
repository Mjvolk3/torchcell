---
id: 622ot8f0s1thwmmqak8bl9c
title: Build_splits
desc: ''
updated: 1795561802556
created: 1795561802556
---

## 2026.11.24 - Pinned splits

Writes `torchcell/tasks/data/<task>.splits.csv.gz` and `results/splits_summary.csv`. Single-gene tasks get 14 splits (random x5, identity 30 % x3, identity 50 % x3, position x3); the pair task gets 17 (random x5, four gene-holdout families x3).
