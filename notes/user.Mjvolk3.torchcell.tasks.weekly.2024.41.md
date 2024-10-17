---
id: p5c3e0lblbs5ycsjspvi525
title: '41'
desc: ''
updated: 1729182475496
created: 1728315119855
---
## 2024.10.07

- [x] [[torchcell.datamodules.cell]] should create indices just based off what it inherits from the dataset. It should not have to iterate over data.
- [x] Add a method to summarize split details. → this works nice for now.
- [x] Contacted [[ncsa.help]] about `vm` activation
- [x] Create [[torchcell.datamodules.perturbation_subset]] to make sure that we have all single perturbations, and then equal counts of 2 and 3 from there. with increasing dataset sizes. We should be able to use the `split_index` for this. → It looks pretty good for now.

- [x] Create dataset subsets. `1e4`, `5e4` ,`1e5`, `5e5` ,`1e6`, and `all`. → hve temp version
- [x] Create tables of subsets.
- [x] Create plots of subsets.

- [x] Write DiffPool Model. Match model hyperparameters to DCell. Make sure all intermediates includes attention and pool mapping are returned and can be inspected for interpretability. → not yet parameter matched
- [x] Write Training Script for DiffPool
- [x] How to handle `None` label → masked loss
- [x] Run training on small data. Set up with data parallel in mind. → ddp not yet checked
- [x] Run test on `Delta`. → failed due to dependency on `raw_db`
- [x] Setup for training on `1e4` and choose reasonable number of hyper parameters for sweep. Launch sweep. Try to avoid data parallel on this and sweep with multiple GPUs. Probably don't use batch size as one of params. → need to refactor first.

## 2024.10.09

- [x] [[Dataset_index_split|dendron://torchcell/experiments.003-fit-int.dataset_index_split]]
- [x] Get a `DeepSet` model working first.
- [x] Message passing with `DiffPool` → working model but needs revision
- [x] Create regression task

## 2024.10.10

- [x] apply skip gat to all gat layers.
- [x] [[2024.10.10 - Fixing Install on Rocky Linux 9|dendron://torchcell/python.lib.torch-scatter#20241010---fixing-install-on-rocky-linux-9]]
- [x] permanently fix mount with fstab
- 🔲 Write function to get distributions over different dataset sizes... use this to prepare traditional machine learning datasets.

## 2024.10.11

- 🔲 Move raw out of init [[torchcell.data.neo4j_cell]] causing issue with loading.
- 🔲 Need nan on plotting find in `5135675`
