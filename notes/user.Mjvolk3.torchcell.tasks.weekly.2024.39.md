---
id: eml0ufuhov06z54rkakmhpj
title: '39'
desc: ''
updated: 1727811991828
created: 1727139041563
---
## 2024.09.23

- [x] Killed `GH` build do to coin miner hack
- [x] Wrote [[Conversion|dendron://torchcell/torchcell.datamodels.conversion]] → seems reasonable but not tested.
- [x] Gene essentiality conversion to fitness [[Gene_essentiality_to_fitness_conversion|dendron://torchcell/torchcell.datamodels.gene_essentiality_to_fitness_conversion]]
- [x] Refactor [[torchcell.data.deduplicate]] → Seems reasonable not tested.
- [x] Refactored [[torchcell.data.neo4j_cell]] so we can have a way of iterating through process steps, and optionally saving intermediates for debugging.

## 2024.09.24

- [x] Solved some import errors with conditional typing checking. → [[torchcell.datamodels.gene_essentiality_to_fitness_conversion]], [[Conversion|dendron://torchcell/torchcell.datamodels.conversion]]
- [x] [[torchcell.datamodels.conversion]] works
- [x] [[torchcell.data.deduplicate]] works
- [x] Refactor [[torchcell.data.deduplicate]] to make it more general.
- [x] Should we add the possibility for multiple deduplicators? There are some forms of deduplication that are not guaranteed to eliminate not a function error? → #ramble Turns out we don't need to do this because the only layer that matters is the first layer of attr. These are all that needs to be the same and then we should be able to put on any object. This means we can put on the entire experiment. The current plan is to have a generic `process_graph` that just adds the entire experiment. If this is too much data we can always use more specialized versions that cut out data.
- [x] Test out conversion and deduplication with `smf` queries essentiality.

## 2024.09.25

- [x] Email Cheng
- [x] Email Ge
- [x] Email Rana
- [x] [[2024.09.25 - Dataset Name on Experiment Reference|dendron://torchcell/torchcell.datamodels.schema#20240925---dataset-name-on-experiment-reference]]
- [x] #ramble I think the issue with putting pydantic model on graph is that it cannot be transferred to gpu. → this is true... this means we need a generic way of converting the experiments
- [x] pass a `process_graph` function in the initializer. It should return with standard types. → returns a `HeteroData` obj.
- [x] Write Aggregator to create list of `experiments` or `phenotype`
- [x] Test out larger `smf` and `essentiality` query
- [x] Run query on `DmfCostanzo2016` and `DmiCostanzo2016` to see if we get dataset.

## 2024.09.26

- [x] This should take in `Neo4jCellDataset` and return `Neo4jCellSubsetDataset`. We can use any of the indices to make this subsetting.→ This is what I wrote earlier... "write function to subset dataset. We want to do this based off of `label_name`, `label_statistic_name`, `dataset`". → I think it is fine to have a class initialized by a `Neo4jCellDataset` that runs subsetting methods according to available indices. → Still deciding how to handle this.

## 2024.09.27

- [x] Slides


