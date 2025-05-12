---
id: yorbzjso3ly821r3lbmf5h0
title: '19'
desc: ''
updated: 1746844664374
created: 1746570667745
---
## 2025.05.05

- [x] Dango Model → minimal progress. Realized issue with data formatting
- [x] Started [[Sahand-Shafeei.preliminary-project-outline.2025.05.05|dendron://torchcell/user.Sahand-Shafeei.preliminary-project-outline.2025.05.05]]

## 2025.05.06

- [x] [[2025.05.06 Meeting|dendron://torchcell/user.Sahand-Shafeei.preliminary-project-outline.2025.05.05#20250506-meeting]]
- [x] `follow_batch` for following perturbations indices for indexing.

## 2025.05.07

- [x] `Dango` running. → still not working well on batch size 64...
- [x] Adjust transforms with new data format.. → Started this with adding [[Regression_to_classification_coo|dendron://torchcell/torchcell.transforms.regression_to_classification_coo]] and [[Test_regression_to_classification_coo|dendron://torchcell/tests.torchcell.transforms.test_regression_to_classification_coo]]
- [x] Setup experiment → To get things to work had to modify plotting to handle variable targets. Latent embedding of `z_p` doesn't apply since dimensinos are very different.

## 2025.05.08

- [x] Add `String11` to properly estimate `lambda`. Add script for lambda determination and put in config. → [[Dango_lambda_determination|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.dango_lambda_determination]]
- [x] Dango with lambda computation.
- [x] Pass `num_heads` into [[Dango|dendron://torchcell/torchcell.models.dango]]
- [x] Check differences in overfitting for different loss variants. [[2025.05.08 - OverFitting With Different Losses|dendron://torchcell/torchcell.models.dango#20250508---overfitting-with-different-losses]]

## 2025.05.09

- [x] [[Igb Sync Log|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.igb-sync-log]], [[Delta Sync Log|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.delta-sync-log]], [[Gilahyper Sync Log|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.gilahyper-sync-log]]
- [x] Check all are synced from [[2025.05.09 - After long break|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.igb-sync-log#20250509---after-long-break]]
- [x] Launch 2 runs → Since there are 3 different runs and I want to know how helpful it is to have pretraining we need last run. → Submitted last third run. It is training very fast so all will be finished by morning.
- [x] Sync works across Delta and Igb [[2025.05.09 - Testing After Change For Delta|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.igb-sync-log#20250509---testing-after-change-for-delta]]
- [x] Check that [[Int_dango|dendron://torchcell/torchcell.trainers.int_dango]] records weighted losses from [[Dango|dendron://torchcell/torchcell.losses.dango]]. → Great this is already done.

- [ ] DCell?

- [ ] Add morphology. Only Safari browser works. Respond to maintainers about solved problem of downloading database. Might want to store a backup.

- [ ] Morphology Random Forest Baseline

- [ ] Morphology animation ? for fun...

- [ ] HeteroCell?

- [ ] Contrastive DCell head on HeteroCell.

- [ ] sync scripts for delta

***

- [ ] Add Gene Expression datasets
- [ ] For @Junyu-Chen consider reconstructing $S$? Separate the metabolic graph construction process into building both $S$ then casting to a graph... Does this change properties of S? You are changing the constrains but also changing the dimensionality of the matrix... → don't know about this... I think the the constrained based optimization won't be affected from the topological change much. It is mostly a useful abstraction for propagating genes deletions.

- [ ] #ramble Need to start dumping important experimental results into the experiments folder under `/experiments` - Do this for `004-dmi-tmi` that does not work

- [ ] Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] Inquiry about web address for database.

- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
-
