---
id: ha7mvvbdznrjbaih4ioyhs4
title: '05'
desc: ''
updated: 1738526735158
created: 1737999272613
---
## 2025.01.27

- [x] Implement isomorphic cell. Fix Metabolism Processor. → wip
- [x] [[2025.01.27 - Metabolism Processor|dendron://torchcell/torchcell.models.isomorphic_cell#20250127---metabolism-processor]]
- [x] [[2025.01.27 - Metabolism Processor Algorithm|dendron://torchcell/torchcell.models.isomorphic_cell#20250127---metabolism-processor-algorithm]]
- ![](./assets/images/level_sets_epistasis_2025-01-27-18-03-50.png)
- ![](./assets/images/3d_epistasis_scatter_2025-01-27-18-03-45.png)

## 2025.01.28

- [x] Implement isomorphic cell. Fix Metabolism Processor. → had to change data formatting in [[torchcell.data.neo4j_cell]] converted dicts to hyperedge indices for easier batching processing in algorithm. → got to point where we discovered improper data handling that was showing CUDA error. Likely issue is always on data side when there is CUDA error when using `pyg`. Had to switch to simpler attention mechanism to troubleshoot this.

## 2025.01.29

- [x] [[Isomorphic_cell_attentional|dendron://torchcell/torchcell.models.isomorphic_cell_attentional]] → [AttentionalAggr](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.AttentionalAggregation.html)
- [x] StoichiometricGraphConv Gating Ideas - [[175438|dendron://torchcell/scratch.2025.01.29.175438]] → [[Stoichiometric_hypergraph_conv|dendron://torchcell/torchcell.nn.stoichiometric_hypergraph_conv]]
- [x] Implement `SupCR` loss. → [[SupCR|dendron://torchcell/torchcell.losses.SupCR]]

## 2025.01.30

- [x] I am uncertain of dist loss we should cook up simple unrelated benchmark to see if it is working. Let's also add in `SupCR`. → build script to test this.
- [x] Create is_any index for gene contained in sets of genes in an instance.

## 2025.01.31

- [x] Define sweep that can finish in < 40 hours.
- [x] Make sure `Qm9` we have all correct plotting.
- [x] Run `Qm9` small test. → running GH → works move to Delta.

- [x] Create metabolism subset dataset. Add to index? → I thought maybe we could push it into query but we can't because the `gene_set` needs to be a superset of all possible genes queried . We need to push it into CellDatasetModule, then run Perturbation subset on that. → Turns out just needs to be in `PerturbationSubset`. → haven't run yet. takes time.
- [x] Plots of dataset breakdown. → [[experiments.003-fit-int.scripts.metabolism_enriched_perturbation_subset]]

- [x] Run all [[Metabolism_enriched_perturbation_subset|dendron://torchcell/experiments.003-fit-int.scripts.metabolism_enriched_perturbation_subset]] overnight.

## 2025.02.01

- [x] Write total loss function
- [x] Test new loss function.
- [x] Overfit batch
- [x] Metabolically enriched data. → note with plots.
