---
id: b26rqf148bt4oze1ou7asv3
title: '29'
desc: ''
updated: 1752609811250
created: 1752608927061
---
- [ ] For @Junyu-Chen consider reconstructing $S$? Separate the metabolic graph construction process into building both $S$ then casting to a graph... Does this change properties of S? You are changing the constrains but also changing the dimensionality of the matrix... → don't know about this... I think the the constrained based optimization won't be affected from the topological change much. It is mostly a useful abstraction for propagating genes deletions.
- [ ] #ramble Need to start dumping important experimental results into the experiments folder under `/experiments` - Do this for `004-dmi-tmi` that does not work
- [ ] Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] Inquiry about web address for database.
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] HeteroCell on String 12.0
- [ ] Contrastive DCell head on HeteroCell.
- [ ] Morphology Random Forest Baseline
- [ ] Morphology animation ? for fun...

***

## 2025.07.15

- [x] Summarize experiments from last week and weekend → The main findings from the experiments was that dropout at any probability would tank the overfitting which makes some sense, but it also tanks performance on large runs. We can regularize in other ways. The other realization is that the `dist` and `supCR` losses need larger batch sizes. This lead to [[Mle_dist_supcr|dendron://torchcell/torchcell.losses.mle_dist_supcr]] where we are trying to abstract the different elements and then achieve larger batch size with different strategies. 1. Accumulated gradients, 2. Circular buffer for stale batch input. 3. gpu gather across DDP.  

![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-09-00-42-19/training_epoch_1999.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-09-07-44-54/training_epoch_0635.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-09-11-02-19/training_epoch_0903.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-09-22-23-12/training_epoch_0099.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-09-23-32-15/training_epoch_0103.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-10-00-45-28/training_epoch_1257.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-10-14-58-09/training_epoch_0121.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-10-15-09-52/training_epoch_0205.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-10-15-30-18/training_epoch_0213.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-10-15-51-24/training_epoch_0303.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-10-19-10-41/training_epoch_0739.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-10-20-33-12/training_epoch_0601.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-10-21-40-12/training_epoch_2265.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-11-01-44-12/training_epoch_4751.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-11-23-42-46/training_epoch_5000.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-12-16-50-57/training_epoch_0911.png)
![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-07-12-18-28-36/training_epoch_0577.png)

- [ ] Notes commit

- [ ] Get new loss working

***

- [ ] Move this into box /Users/michaelvolk/Documents/projects/torchcell/data/host/kuzmin2018/aao1729_data_s1.zip

- [ ] Add morphology. Only Safari browser works. Respond to maintainers about solved problem of downloading database. Might want to store a backup.
- [ ] Add Gene Expression datasets

- [ ] Aggregate results from Dango 005 and put into table.
- [ ] Run Dango on `006` using best settings from `005`
