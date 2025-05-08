---
id: ql5qqwglqumgcpu5imec7bz
title: '18'
desc: ''
updated: 1746570786873
created: 1745956797334
---

## 2025.04.28

- [x] Get querying working. - [[005-kuzmin2018-tmi.query.issue|dendron://torchcell/experiments.005-kuzmin2018-tmi.query.issue]] → Had issue that some of the data wasn't included in previous query of Kuzmin2018 due to temp sensitive alleles.

## 2025.04.29

- [x] Added summary table. [[String_vs_sgd_vs_tflink|dendron://torchcell/experiments.004-dmi-tmi.scripts.string_vs_sgd_vs_tflink]]
- [x] string9 graph to selective only load `String9` graphs → [[torchcell.graph.graph]]
- [x] Update `torchell.graph` adding in types for `GeneGraph` and `MultiGeneGraph` this will save us headache in `graph_processor.`

## 2025.04.30

- [x] Add Morphology datasets → figured out download..

- [x] Make sure that `SubgraphRepresentation` can handle the updated data format → Just using `('gene', . ,'gene')` for gene perturbation.
- [x] Update loading s.t. it accounts for `String9` → Things look good

## 2025.05.01

- [x] Only include necessary graphs in [[Load_batch_005|dendron://torchcell/torchcell.scratch.load_batch_005]] → Think issue is handing no `incidence_graphs`.
- [x] Fix subgraph representation such that it can handle any graph from multigraph
- [x] Implement Dango model → got Dango pretraining done.

## 2025.05.03

- [x] Since we don't need to subgraph we are using `Perturbation(GraphProcessor)`. Modified to coo. → Updated [[torchcell.models.dango#data|dendron://torchcell/torchcell.models.dango#data]] and [[Load_batch_005|dendron://torchcell/torchcell.scratch.load_batch_005]]
- [x] Implement Dango. → started.
