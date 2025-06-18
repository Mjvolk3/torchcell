---
id: j7ag2z1rgofx48iryne1mts
title: '24'
desc: ''
updated: 1750221887941
created: 1749503722566
---

- [x] Implement transforms. → [[2025.04.02 - Named Phenotype Labelling|dendron://torchcell/torchcell.models.hetero_cell_bipartite#20250402---named-phenotype-labelling]]
- [x] Make sure tests pass.
- [x] Run model locally to check if running.
- [x] Check logging of data.
- [x] DDP run.
- [x] Make sure models are saving.
- [x] #ramble Metabolic genes output. Can run comparative flux variability analysis or sampling to see try to (1) explain why cells grow faster as it relates to metabolism. (2) Find triple mutants that would have higher flux through a particular pathway. Engineering objective. Identifying a triple mutant that would grow faster synergistically while improving flux through a particular pathway. Need to down select a set of market relevant intermediates that are easy to detect via HPLC.

## 2025.06.09

- [x] Run 006 query. → Had to set graphs to null → 332,313 trigenic interactions
- [x] Sync experiments for `005`, see if syncing crashes offline runs. Fastest run is on epoch 153.  → [[2025.06.03 - Launched Experiments|dendron://torchcell/experiments.005-kuzmin2018-tmi.results.hetero_cell_bipartite_dango_gi#20250603---launched-experiments]] jobs stopping will also be sign of cancellation. Keeping `slurm id: 1820177`, cancelling `slurm id: 1820176`, `slurm id: 1820165`. → Can sync during runs without issue.
- [x] After sync check if the gating of loss looks dramatic. If so we should parameterize concatenation. → gating looks worse [[2025.06.17 - I Cannot Find Pattern Of Jumping Correlation|dendron://torchcell/experiments.005-kuzmin2018-tmi.results#20250617---i-cannot-find-pattern-of-jumping-correlation]]
- [x] Debugged ICLoss NaN issue with hetero_cell_bipartite_dango_gi

1824013 - lr: 1e-4
lr: 1e-5

## 2025.06.10

- [x] Fix issue with ddp on genome. → Implemented file locking solution with fcntl for all cached JSON properties
- [x] [[ICLoss NaN Issue Summary (2025.06.10)|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.hetero_cell_bipartite_dango_gi#icloss-nan-issue-summary-20250610]]

## 2025.06.11

- [x] [[2025.06.12 - Showing Values for Data When COO Format|dendron://torchcell/torchcell.scratch.load_batch_005#20250612---showing-values-for-data-when-coo-format]]
- [x] `.claude/commands/coo_transforms.md`

## 2025.06.12

- [x] [[2025.06.10 - Issue Causing Nans in Z_W|dendron://torchcell/torchcell.losses.isomorphic_cell_loss#20250610---issue-causing-nans-in-z_w]]
- [x] Sync [[2025.06.10 - HeteroCell Bipartite Dango GI Single GPU DDP Not Working Yet|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.igb-sync-log#20250610---heterocell-bipartite-dango-gi-single-gpu-ddp-not-working-yet]] → [[2025.06.12 - HeteroCell Bipartite Dango ICLoss DDP|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.igb-sync-log#20250612---heterocell-bipartite-dango-icloss-ddp]]
- [x] [[2025.06.12 - DCell 005-Kuzmin2018-tmi|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.igb-sync-log#20250612---dcell-005-kuzmin2018-tmi]]

## 2025.06.15

- [x] Looks like the transforms don't help at all... This actually makes some sense. Can we know this a priori? Trying another experiment with dist loss cranked up.

```bash
(torchcell) mjvolk3@biologin-2 ~/projects/torchcell $ sbatch /home/a-m/mjvolk3/scratch/torchcell/experiments/005-kuzmin2018-tmi/scripts/igb_optuna_hetero_cell_bipartite_dango_gi-ddp.slurm
Submitted batch job 1826116
(torchcell) mjvolk3@biologin-2 ~/projects/torchcell $ squeue -p cabbi
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1826088     cabbi     HCPD  mjvolk3  R   15:40:20      1 compute-3-3
           1826116     cabbi     HCPD  mjvolk3  R       0:01      1 compute-3-3
(torchcell) mjvolk3@biologin-2 ~/projects/torchcell $ squeue -p cabbi
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1826088     cabbi     HCPD  mjvolk3  R   15:43:19      1 compute-3-3
           1826116     cabbi     HCPD  mjvolk3  R       3:00      1 compute-3-3
```

Looks like the we randomly get Pearson curves up to 0.2 and others up to 0.5. It doesn't look like theres

[[2025.06.17 - I Cannot Find Pattern Of Jumping Correlation|dendron://torchcell/experiments.005-kuzmin2018-tmi.results#20250617---i-cannot-find-pattern-of-jumping-correlation]]
