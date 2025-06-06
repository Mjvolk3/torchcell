---
id: tujklzpdde0fdscufqli94n
title: '23'
desc: ''
updated: 1749087275616
created: 1748899404988
---
## 2025.06.02

- [x] Clean up [[tasks.weekly.2025.21|dendron://torchcell/user.Mjvolk3.torchcell.tasks.weekly.2025.21]]
- [x] Simplify embedding config embedding
- [x] `hetero_cell` all graphs with String 12, naive pred head → Had to update batch loading → [[2025.06.02 - Data Updated|dendron://torchcell/torchcell.models.hetero_cell_bipartite_dango#20250602---data-updated]]

## 2025.06.03

- [x] `hetero_cell_bipartite_dango_gi`  → [[2025.06.03 - Data Updated|dendron://torchcell/torchcell.models.hetero_cell_bipartite_dango_gi#20250603---data-updated]] → model working
- [x] [[2025.06.03 - Detailed View of Data For Indexing Based on Phenotype Type|dendron://torchcell/torchcell.models.hetero_cell_bipartite_dango_gi#20250603---detailed-view-of-data-for-indexing-based-on-phenotype-type]]
- [x] Launch model with all graphs on igb biocluster. [[2025.06.03 - Launched Experiments|dendron://torchcell/experiments.005-kuzmin2018-tmi.results.hetero_cell_bipartite_dango_gi#20250603---launched-experiments]]

## 2025.06.04

- [x] Icloss support in [[torchcell.models.hetero_cell_bipartite_dango_gi]]
- [x] Made local prediction more dango like. →  We are seeing global dominate. ![](./assets/images/hetero_cell_bipartite_dango_gi_training_2025-06-04-19-49-43/training_epoch_0221.png) overfitting looking good.

- [ ]

- [ ] Query 006 - All trigenic interaction data.
- [ ] `hetero_cell_bipartite_dango` - fitness pred

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
- [ ] HeteroCell on String 12.0
- [ ] Contrastive DCell head on HeteroCell.
- [ ] Add morphology. Only Safari browser works. Respond to maintainers about solved problem of downloading database. Might want to store a backup.
- [ ] Morphology Random Forest Baseline
- [ ] Morphology animation ? for fun...
