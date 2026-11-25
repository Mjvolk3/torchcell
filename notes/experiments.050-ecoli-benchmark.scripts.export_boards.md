---
id: o7ym1d0raw6t5jofcx7e4q4
title: Export_boards
desc: ''
updated: 1795573647064
created: 1795573647064
---

## 2026.11.24 - Board export

Writes one board directory per task and seed-0 split (splits, labels, and every linear-probe submission) for the benchmark service on the website branch, where `experiments/035-benchmark-bundles/scripts/ecoli_task_bundles.py` turns each into a bundle. 13 boards were loaded into the STAGING service on 2026-11-24 with 172 baseline rows (growth rate 4 boards x 9, imaging 4 x 9, genetic interactions 5 x 20). The metabolome task is not exported: 7,534 targets is about 5.4 million prediction rows per upload against a 16 MB limit. Morphology feature names are rewritten for the board (`<L>` to `mean_L`, `%2N` to `pct_2N`).
