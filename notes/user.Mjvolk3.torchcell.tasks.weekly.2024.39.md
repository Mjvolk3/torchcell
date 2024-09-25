---
id: eml0ufuhov06z54rkakmhpj
title: '39'
desc: ''
updated: 1727224974888
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
- [ ] Test out conversion and deduplication with `smf` queries essentiality.
- [ ] Write Aggregator
- [ ] pass a `process_graph` function in the initializer. It should return with standard types.

***

- [ ] Add in conversion with simple example first. Use essential to fitness.`ABC` should take in a raw dataset and return a conversion map.
- [ ] Add possibility for multiple deduplicators... deduplicators should be broadly broken into two classes those that guarantee functional mapping, `mean`, `max` type deduplication and those that don't have this guarantee.
- [ ] Change "interaction" to "gene_interaction"
- [ ] Rebuild local database
- [ ] Rebuild large database

- [ ] Create `1e03`, `1e04`, and `1e05` datasets with positive `tmi`. → This will be difficult because it'll be hard to balance mutant types. We could just use triple mutants with the plan to down select by enriched double mutants.
