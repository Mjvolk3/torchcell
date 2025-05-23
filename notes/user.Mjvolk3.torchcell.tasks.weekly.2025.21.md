---
id: byqxlgenke415eplp31l2fk
title: '21'
desc: ''
updated: 1748034154361
created: 1747592127726
---

## 2025.05.19

- [ ] Dcell with strata.
- [ ]

***

- [ ] HeteroCell on String 12.0
- [ ] Contrastive DCell head on HeteroCell.

- [ ] Add morphology. Only Safari browser works. Respond to maintainers about solved problem of downloading database. Might want to store a backup.
- [ ] Morphology Random Forest Baseline
- [ ] Morphology animation ? for fun...

## 2025.05.20

- [x] [[2025.05.20 - Investigatin DCell Absurdly Slow Iteration|dendron://torchcell/torchcell.models.dcell#20250520---investigatin-dcell-absurdly-slow-iteration]]
- [x] [[2025.05.20 - DCell with Strata|dendron://torchcell/torchcell.scratch.load_batch_005#20250520---dcell-with-strata]]
- [ ] Get `DCell` to work on device.

**BATCH SIZE = 32**

On Delta GPU:

`Loss: 0.001867, Corr: 0.4769, Time: 100.151s/epoch:  62%|███  62/100 [1:44:31<1:04:03, 101.15s/it]`

On M1 CPU:

Much faster. This is when we realized we need some other solution.

`20s/it`

## 2025.05.21

- [x] [[2025.05.21 - Data Structure for Pyg Message Passing|dendron://torchcell/torchcell.scratch.load_batch_005#20250521---data-structure-for-pyg-message-passing]]
- [x] [[2025.05.21 - Data Structure Don't Propagate Edge Info|dendron://torchcell/torchcell.scratch.load_batch_005#20250521---data-structure-dont-propagate-edge-info]]
- [x] Revert `DCell` I think we can make this better with mutant state matrix. Get mutant state right.
- [x] Model initialization.

## 2025.05.22

- [x][[2025.05.23 - One DCell Module Processing|dendron://torchcell/torchcell.models.dcell#20250523---one-dcell-module-processing]]
- [x] Get `Dcell` working. [[2025.05.22 - DCell overfit on M1|dendron://torchcell/torchcell.models.dcell#20250522---dcell-overfit-on-m1]]

## 2025.05.23

- [x] Experiment without alpha. [[2025.05.23 - DCell overfit on M1 without alpha|dendron://torchcell/torchcell.models.dcell#20250523---dcell-overfit-on-m1-without-alpha]]
- [ ] Consolidate and commit.
- [ ] Test speed on GPU.

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
