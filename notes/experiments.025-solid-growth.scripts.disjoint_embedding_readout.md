---
id: j8skj8tu6dy4vc10tqc3vpa
title: Disjoint_embedding_readout
desc: ''
updated: 1789319071191
created: 1789319071191
---

## 2026.09.13 - Readout of the disjoint-split sequence-embedding arms

`experiments/025-solid-growth/scripts/disjoint_embedding_readout.py` selects each arm's
runs from W&B by the config-name tag the run script attaches (`cgt_s0_q_kl_*`), keeps the
rank-0 run (the one carrying `val/gene_interaction/Pearson`), and writes one row per
config to `experiments/025-solid-growth/results/disjoint_embedding_readout.csv`: max
validation Pearson and its epoch (a biased order statistic), the value at epoch 29 (the
protocol's fixed reading), the mean over epochs 10 to 29 (a window average), the mean over
60 to 99 for the 100-epoch runs, validation point loss at 29 and at the last epoch, train
Pearson and the graph penalty at the last epoch. It prints a markdown table and the run
URLs. Rerun after any sync; a run's `complete` flag says whether it reached its budget.

The findings read from it are in
[[experiments.025-solid-growth.scripts.equivariant_cell_graph_transformer]] under the
same date.
