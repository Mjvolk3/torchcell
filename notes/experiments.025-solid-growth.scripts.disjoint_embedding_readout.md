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

### Grouped Charts view

`disjoint_embedding_wandb_view.py` renames every disjoint-split run `<arm>_seed<k>`
(`_rank<r>` for the three non-rank-0 runs of a job), writes the config keys `arm`,
`seed_`, `split`, `rank0`, and overwrites the saved workspace view
`025 disjoint split: arms grouped` (id `paezdq4q5ex`), runset grouped by `arm`, four
sections in rank order: held-out query pairs, training side, operator and probe,
bookkeeping. Rerun after every sync:
<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer?nw=paezdq4q5ex>

## 2026.09.16 - The last two mmli cells synced: composite at three seeds, composite plus fitness at one

IGB mmli jobs 2397848 (`cgt_s0_q_kl_emb_017` seed 2, finished 02:01 CDT) and 2397876 (`cgt_s0_q_kl_embfit_027` seed 42, finished 10:03 CDT) synced from the login node and folded into `results/disjoint_embedding_readout.csv` and the grouped view (55 runs labeled). The mmli GPUs are now held by another user's array, so nothing of ours is queued there.

| arm | seed | val Pearson mean ep 10-29 | max (epoch) | epoch 29 |
|---|---|---|---|---|
| composite (emb_017) | 1 / 2 / 42 | 0.235 / 0.203 / 0.215 | 0.275 (15) / 0.270 (2) / 0.263 (2) | 0.173 / 0.170 / 0.222 |
| composite + fitness (embfit_027) | 42 | 0.209 | 0.254 (2) | 0.158 |
| learnable table (ctrl_016) | 1 / 2 / 42 | 0.144 / 0.125 / 0.140 | 0.226 / 0.163 / 0.195 | 0.136 / 0.123 / 0.130 |

The composite arm's three-seed window mean is 0.218 against the control's 0.136; the joint fitness objective at one seed (0.209, fitness val Pearson max 0.730) sits inside the composite arm's seed spread and does not move the interaction score. Grouped view: `?nw=paezdq4q5ex`.

## 2026.09.17 - Composite plus fitness at three seeds

mmli 2408604 (seed 1, ended 01:43) and 2408605 (seed 2, ended 11:12) synced; 63 runs labeled in the grouped view. Window mean of validation Pearson over epochs 10 to 29, paired by seed against the composite arm without the fitness head:

| seed | composite (emb_017) | composite + fitness (embfit_027) | difference |
|---|---|---|---|
| 1 | 0.235 | 0.237 | +0.002 |
| 2 | 0.203 | 0.172 | -0.032 |
| 42 | 0.215 | 0.209 | -0.006 |
| mean | 0.218 | 0.206 | -0.012 |

Two of three seeds below, one level; the mean difference is inside the across-seed spread (sd 0.016 on the composite arm), so the fitness objective neither helps nor measurably costs the interaction score on the disjoint split. Seed 1 of the joint arm holds 0.264 at epoch 29 with its maximum at epoch 25, the only cell of either arm whose curve did not decay after epoch 15. Fitness validation Pearson max 0.730 (seed 42); seeds 1 and 2 to be read from the same column.
