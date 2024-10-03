---
id: h59o9thoyum3pqfxrxqbrcu
title: '40'
desc: ''
updated: 1727992376528
created: 1727811879911
---
## 2024.10.01

- [x] Composition converter. → We have a decent composite strategy for now [[torchcell.datamodels.fitness_composite_conversion]]
- [x] Handle `None` in deduplicate. → We seem to already be handing `None` fine. → [[torchcell.data.mean_experiment_deduplicate]]
- [x] Unique naming of statistic field. → we also did not remove `dataset_name` from the reference like we were thinking of doing because the uniqueness of the reference would be maintained without the `dataset_name`. References can be converted to be unified.
- [ ] Rebuild database.

## 2024.10.03

- [ ] GH

## 2024.10.03

- [ ] Synthetic Lethality phenotypes not linked to query. Breaks `PhenotypeMemberOf` relation. → Checked two of these broken edge types.
- [ ] `SynthLethalityYeastSynthLethDbDataset` is a missing node. Breaks `ExperimentMemberOf` relation. 
- [ ] `Mentions` relation is broken. All the `to` and `from` look mostly different and this is really only in the case of a secondary data source like synthetic lethality.

***

- [ ] Ran query after fixing
- [ ] Email Exxact

- [ ] Write `1e3`, `1e4`, `5e4` ,`1e5`, `5e5` ,`1e6`, and `all`.
- [ ] Check data distributions for all data.

- [ ] Write DiffPool Model. Match model hyperparameters to DCell. Make sure all intermediates includes attention and pool mapping are returned and can be inspected for interpretability.
- [ ] Write Training Script for DiffPool
- [ ] How to handle `None` label
- [ ] Run training on small data. Set up with data parallel in mind.
- [ ] Run test on `Delta`.

- [ ] Setup for training on `1e4` and choose reasonable number of hyper parameters for sweep. Launch sweep. Try to avoid data parallel on this and sweep with multiple GPUs. Probably don't use batch size as one of params.

- [ ] Add naming to statistic label, then rebuild db.

- [ ] `GH` rebuild. Get art first.
- [ ] Random forest regression can be used for multi-label prediction

- [ ] Add synthetic lethal to dataset.
- [ ] Figure out data subsetting. → This is a bit more difficult than I anticipated. I am not sure if I should do this at the dataset or data module level.

- [ ] Email Illinois Computes
- [ ] Add in conversion with simple example first. Use essential to fitness.`ABC` should take in a raw dataset and return a conversion map.
- [ ] Add possibility for multiple deduplicators... deduplicators should be broadly broken into two classes those that guarantee functional mapping, `mean`, `max` type deduplication and those that don't have this guarantee.
- [ ] Change "interaction" to "gene_interaction"
- [ ] Rebuild local database
- [ ] Rebuild large database

- [ ] Create `1e03`, `1e04`, and `1e05` datasets with positive `tmi`. → This will be difficult because it'll be hard to balance mutant types. We could just use triple mutants with the plan to down select by enriched double mutants.
