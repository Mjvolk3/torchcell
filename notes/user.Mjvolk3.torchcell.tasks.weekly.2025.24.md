---
id: j7ag2z1rgofx48iryne1mts
title: '24'
desc: ''
updated: 1749505527275
created: 1749503722566
---
## 2025.06.09

- [x] Run 006 query. → Had to set graphs to null → 332,313 trigenic interactions
- [ ] Sync experiments for `005`, see if syncing crashes offline runs. Fastest run is on epoch 153.  → [[2025.06.03 - Launched Experiments|dendron://torchcell/experiments.005-kuzmin2018-tmi.results.hetero_cell_bipartite_dango_gi#20250603---launched-experiments]] jobs stopping will also be sign of cancellation. Keeping `slurm id: 1820177`, cancelling `slurm id: 1820176`, `slurm id: 1820165`. →
- [ ] After sync check if the gating of loss looks dramatic. If so we should parameterize concatenation.  
- [ ] Run Dango on `006` query
- [ ]
- [ ]

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
