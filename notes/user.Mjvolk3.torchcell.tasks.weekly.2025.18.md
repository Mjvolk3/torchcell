---
id: ql5qqwglqumgcpu5imec7bz
title: '18'
desc: ''
updated: 1746147705272
created: 1745956797334
---

## 2025.04.28

- [x] Get querying working. - [[005-kuzmin2018-tmi.query.issue|dendron://torchcell/experiments.005-kuzmin2018-tmi.query.issue]] → Had issue that some of the data wasn't included in previous query of Kuzmin2018 due to temp sensitive alleles.

## 2025.04.29

- [x] Added summary table. [[String_vs_sgd_vs_tflink|dendron://torchcell/experiments.004-dmi-tmi.scripts.string_vs_sgd_vs_tflink]]
- [x] string9 graph to selective only load `String9` graphs → [[torchcell.graph.graph]]
- [x] Update `torchell.graph` adding in types for `GeneGraph` and `MultiGeneGraph` this will save us headache in `graph_processor.`

- [ ] Make sure that SubgraphRepresentation can handle the

- [ ] Update loading s.t. it accounts for `String9` →

- [ ] Fix subgraph representation such that it can handle any graph from multigraph

- [ ] Implement Dango model

## 2025.04.30

- [x] Add Morphology datasets → figured out download..

## 2025.05.01

- [ ]
- [ ] Add Gene Expression datasets
- [ ] 

***

- [ ] For @Junyu-Chen consider reconstructing $S$? Separate the metabolic graph construction process into building both $S$ then casting to a graph... Does this change properties of S? You are changing the constrains but also changing the dimensionality of the matrix... → don't know about this... I think the the constrained based optimization won't be affected from the topological change much. It is mostly a useful abstraction for propagating genes deletions.

- [ ] #ramble Need to start dumping important experimental results into the experiments folder under `/experiments` - Do this for `004-dmi-tmi` that does not work

- [ ] Morphology Random Forest Baseline
- [ ] Morphology animation ?

- [ ] Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] Inquiry about web address for database.

- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
-
