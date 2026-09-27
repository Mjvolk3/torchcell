---
id: vmc691w9qpmmjw53pfdn1wu
title: Graph_reg_sweep_readout
desc: ''
updated: 1790475556572
created: 1790475556572
---

## 2026.09.26 - Reading the graph-regularization sweep out of W&B

Selects runs by the tags the training script attaches (`graph_reg_sweep`, `cgt_s0_r_kl_ctrl_013`), keeps Delta rank-0 runs in state finished or running, and classifies each by its own config: `attention_mask.enabled` is the mask arm, `random_graph.enabled` the random-graph arm, else `graph_reg_lambda` names the ladder point. Failed and crashed runs are launches that died, never measurements; GilaHyper runs of the control (seeds 42, 43) are off-protocol and excluded. 26 runs excluded at the 2026-09-27 pull, 26 kept.

Per run: validation Pearson at epoch 29 (fixed reading) and its max over epochs with the epoch (biased), the window mean over epochs 10 to 29, validation point loss, train Pearson, the graph penalty as logged and divided by lambda (the divergence, comparable across the ladder; the model multiplies by lambda once, per head, and all nine heads share `graph_reg_lambda`), edge recall at degree and precision at k = 32 averaged over the nine regularized heads at the diagnostic epochs (9, 19, 29), and the gradient-probe norms at epochs 0, 1, 2, 5, 10, 20. `val_edge_recovery_summary/*` keys are images, not scalars; the per-graph `recall_at_deg` and `precision_k*` keys are the scalars.

Writes `results/graph_reg_sweep_runs.csv`, `graph_reg_sweep_history.csv` (long), `graph_reg_sweep_summary.json`, and `notes-tex/025-graph-reg-sweep/tables/t1-arms.tex`, `t2-runs.tex`. `--offline` rebuilds the summary and tables from the CSVs; `--regroup` sets each run's W&B group to its arm (the ladder was launched as overrides of `ctrl_013`, whose config set every run's group to `s0_control_30ep`).

Readings: [[experiments.025-solid-growth.graph-reg-sweep]].
