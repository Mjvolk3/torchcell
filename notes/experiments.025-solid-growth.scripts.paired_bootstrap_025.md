---
id: gb3jrgbehhf0idzl3y0ipyf
title: Paired_bootstrap_025
desc: ''
updated: 1789527920508
created: 1789527920508
---

## 2026.09.15 - The arm Q test comparison, paired

GH 1640's epoch 7 checkpoint scored on the arm Q test part (job 1846, `score_cgt_checkpoint_cpu.py`) against the nulls' per-record test predictions on the same 37,791 records, with a paired bootstrap over records (2,000 resamples, percentile interval), the rule the 010 report used for the random-split margin. Result: transformer 0.1392 Pearson (0.109 Spearman); against B1 0.1853 the gain is -0.0461 [-0.0563, -0.0358]; against B2 0.1800 it is -0.0408 [-0.0512, -0.0301]; against B5 seed 0 at 0.1380 it is +0.0012 [-0.0077, +0.0097]. Residual correlations 0.94 to 0.96; prediction correlation with B1 0.64. The script evaluates the prespecified decision: D1 (test Pearson exceeds B1) FAIL, D2 (paired interval excludes zero) FAIL in the direction of the null, D3 (holds over three seeds) NOT RUN. Writes `results/paired_bootstrap_025.json` and the tables `t6-armq-paired.tex`, `t7-decision.tex`.

<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/runs/327csnlk>
