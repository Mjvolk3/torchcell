---
id: i6ym6pzub2i3322i2hvwz51
title: Score_cgt_checkpoint_cpu
desc: ''
updated: 1789369168981
created: 1789369168981
---

## 2026.09.14 - CPU scoring of job 1640's epoch 7 checkpoint on arm Q

`score_cgt_checkpoint_cpu.py` scores a 025 cell graph transformer checkpoint on an arm's held-out parts without a GPU and without the 3.2 TB LMDB, the way `score_010_checkpoints_directly.py` scored the 010 checkpoints: the encoder runs once on the wildtype gene table, and each record reaches the model only as the indices of its three perturbed genes. Three things had to be right. The index space is `sorted(genome.gene_set)`, 6,607 genes, not the build's 4,352 perturbed genes. The readout pools by sum: `PerturbationHead` now defaults to sum and the 025 configs set no `pooling` key, where the 010 checkpoints pooled by mean. And arm Q trained with `transforms.fit_on: train`, so the inverse transform uses the `train_Q` constants of `label_normalization_constants.json`.

The check is the logged validation Pearson at the checkpoint's epoch. A local CPU run over the full validation part (37,705 records, 7.7 s per 1,000 records on 64 threads) gave 0.199653 against the logged 0.199319 at epoch 7, a difference of 0.0003 consistent with fp32 scoring of a bf16-mixed run, the same order as the 010 reproduction. Test was not scored locally; `gh_score_cgt_checkpoint_cpu.slurm` scores validation and test on GilaHyper's CPU queue (32 CPUs, no GPU) and writes `results/cgt_checkpoint_scores_327csnlk.json` and the per-record prediction arrays that the paired bootstrap against B1 needs.

<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/runs/327csnlk>
