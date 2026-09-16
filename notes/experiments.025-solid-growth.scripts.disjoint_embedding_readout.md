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

### W&B report

`experiments/025-solid-growth/scripts/disjoint_embedding_wandb_report.py` publishes a
report with one run set filtered to the `split_Q` tag, the readout table above as
markdown, and epoch curves of validation Pearson, validation point loss, train Pearson,
the graph penalty and the perturbed-CLS strain spread. Each run creates a new version:

<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/reports/025-disjoint-split:-sequence-embeddings-against-the-learnable-table--VmlldzoxNzkyNTI2OA==>

## 2026.09.15 - One row per seed

The readout now writes one row per rank-0 run of each config (one per seed, `seed`
column) and covers the random-vector control (`_018`) and the composite-plus-fitness arm
(`_027`, with `val_fitness_pearson_max`). The report labels each line with its seed.
Republished version:
<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/reports/025-disjoint-split:-sequence-embeddings-against-the-learnable-table--VmlldzoxNzk0MTc0Nw==>
