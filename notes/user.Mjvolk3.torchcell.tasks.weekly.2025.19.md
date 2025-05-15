---
id: yorbzjso3ly821r3lbmf5h0
title: '19'
desc: ''
updated: 1747352207274
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

## 2025.05.10

- [x] [[2025.05.10 DCell Inclusion|dendron://torchcell/torchcell.scratch.load_batch_005#20250510-dcell-inclusion]]
- [x] [[2025.05.10 - Inspecting Data in GoGraph|dendron://torchcell/torchcell.models.dcell_DEPRECATED#20250510---inspecting-data-in-gograph]]
- [x] [[2025.05.10 - DCell Not Compliant With Torch Norms|dendron://torchcell/torchcell.models.dcell_DEPRECATED#20250510---dcell-not-compliant-with-torch-norms]]
- [x] [[2025.05.10 - Gene Perturbations Without GO will Be Improperly Represented|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.batch_005_investigate_pert_size_2#20250510---gene-perturbations-without-go-will-be-improperly-represented]]
- [x] Move data away from using `dicts` → [[2025.05.10 - DCell No Dicts|dendron://torchcell/torchcell.scratch.load_batch_005#20250510---dcell-no-dicts]]
- [x] [[Dango_lambda_determination_string11_0_to_string12_0|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.dango_lambda_determination_string11_0_to_string12_0]]
- [x] [[2025.05.10 - Updated to Show Lambda|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.dango_lambda_determination#20250510---updated-to-show-lambda]]
