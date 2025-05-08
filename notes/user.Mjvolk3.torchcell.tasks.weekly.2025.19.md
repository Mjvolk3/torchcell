---
id: yorbzjso3ly821r3lbmf5h0
title: '19'
desc: ''
updated: 1746734212338
created: 1746570667745
---
## 2025.05.05

- [x] Dango Model → minimal progress. Realized issue with data formatting
- [x] Started [[Sahand-Shafeei.preliminary-project-outline.2025.05.05|dendron://torchcell/user.Sahand-Shafeei.preliminary-project-outline.2025.05.05]]

## 2025.05.06

- [x] [[2025.05.06 Meeting|dendron://torchcell/user.Sahand-Shafeei.preliminary-project-outline.2025.05.05#20250506-meeting]]
- [ ] `follow_batch` for following perturbations indices for indexing.

- [ ] Add morphology. Only Safari browser works. Respond to maintainers about solved problem of downloading database. Might want to store a backup.
- [ ] Morphology Random Forest Baseline

- [ ] Morphology animation ? for fun...

## 2025.05.07

- [x] `Dango` running. → still not working well on batch size 64...
- [x] Adjust transforms with new data format.. → Started this with adding [[Regression_to_classification_coo|dendron://torchcell/torchcell.transforms.regression_to_classification_coo]] and [[Test_regression_to_classification_coo|dendron://torchcell/tests.torchcell.transforms.test_regression_to_classification_coo]]
- [x] Setup experiment → To get things to work had to modify plotting to handle variable targets. Latent embedding of `z_p` doesn't apply since dimensinos are very different.

## 2025.05.08

- [ ] Add `String11` to properly estimate `lambda`. Add script for lambda determination and put in config.

- [ ] Run benchmark training experiments.

- [ ] Morphology?

- [ ] DCell?
- [ ] HeteroCell?

- [ ] Contrastive DCell head on HeteroCell.



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
