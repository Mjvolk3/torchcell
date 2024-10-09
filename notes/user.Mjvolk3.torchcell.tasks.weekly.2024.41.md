---
id: p5c3e0lblbs5ycsjspvi525
title: '41'
desc: ''
updated: 1728340692573
created: 1728315119855
---
## 2024.10.07

- [x] [[torchcell.datamodules.cell]] should create indices just based off what it inherits from the dataset. It should not have to iterate over data.
- [x] Add a method to summarize split details. → this works nice for now.
- [x] Contacted [[ncsa.help]] about `vm` activation
- [x] Create [[torchcell.datamodules.perturbation_subset]] to make sure that we have all single perturbations, and then equal counts of 2 and 3 from there. with increasing dataset sizes. We should be able to use the `split_index` for this. → It looks pretty good for now.

- [ ] Create dataset subsets. `1e4`, `5e4` ,`1e5`, `5e5` ,`1e6`, and `all`.
- [ ] Create tables of subsets.
- [ ] Create plots of subsets.

- [ ] Write DiffPool Model. Match model hyperparameters to DCell. Make sure all intermediates includes attention and pool mapping are returned and can be inspected for interpretability.
- [ ] Write Training Script for DiffPool
- [ ] How to handle `None` label
- [ ] Run training on small data. Set up with data parallel in mind.
- [ ] Run test on `Delta`.
- [ ] Setup for training on `1e4` and choose reasonable number of hyper parameters for sweep. Launch sweep. Try to avoid data parallel on this and sweep with multiple GPUs. Probably don't use batch size as one of params.

***

- [ ] Change `tree` to `blob` on front matter.
- [ ] Email Exxact

- [ ] Check data distributions for all data.

- [ ] Add naming to statistic label, then rebuild db.

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
