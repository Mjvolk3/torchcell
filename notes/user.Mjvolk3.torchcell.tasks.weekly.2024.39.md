---
id: eml0ufuhov06z54rkakmhpj
title: '39'
desc: ''
updated: 1727214171742
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
- [ ] Refactor [[torchcell.data.deduplicate]] to make it more general.
- [ ] Test out essentiality conversion and deduplication with `smf` queries
- [ ] Write Aggregator

***

- [ ] Add in conversion with simple example first. Use essential to fitness.`ABC` should take in a raw dataset and return a conversion map.
- [ ] Add
- [ ] Change "interaction" to "gene_interaction"
- [ ] Rebuild local database
- [ ] Rebuild large database

- [ ] Create `1e03`, `1e04`, and `1e05` datasets with positive `tmi`. → This will be difficult because it'll be hard to balance mutant types. We could just use triple mutants with the plan to down select by enriched double mutants.
