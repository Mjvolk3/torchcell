---
id: xnqyw5xv3ehhaieqrd7hlo9
title: '35'
desc: ''
updated: 1725815941312
created: 1724674431881
---
## 2024.08.26

- [x] Run fitness on random embedding. → As expected we have improved performance with dimension should try max dim which is size of `one_hot`, `>>> x.shape (1000, 6579)`
- [x] Construct `002-dmi-tmi` `1e06` dataset.
- [x] Run `all_dataloader` for `002-dmi-tmi` `1e03` and `1e04` on m1.
- [x] Brainstorming updates to [[torchcell.datamodels.schema]] → [[2024.08.26 - Generic Subclasses Need to Consider Phenotype Label Index|dendron://torchcell/torchcell.datamodels.schema#20240826---generic-subclasses-need-to-consider-phenotype-label-index]]
- [x] Rerun plotting, trying to fill in all fitness data → almost there.
- [x] Create random `6579` `random` for `002-dmi-tmi` `1e3`, `1e4`
- [x] Create random `6579` `random` for `002-dmi-tmi` `1e5` → will finish overnight
- [x] Fix type case to numeric iss ue is [[Svr|dendron://torchcell/experiments.smf-dmf-tmf-001.svr]]
- [x] Fix no cross validation in [[Svr|dendron://torchcell/experiments.smf-dmf-tmf-001.svr]]

- [x] Create scripts for traditional ml `002-dmi-tmi`.
- [x] Revert to `39f8c79e5a93953c240965becba6e0c59bb54026` to create random `6579` `random`. → only have partial interactions. Missing a few `1e05` bc takes long time to compute. Still not sure if necessary.

## 2024.08.27

- [x] Try to compute `random_6579` for traditional. → Couldn't complete. `1e5` `no_pert` had some memory issues.
- [x] Traditional ml on interactions.

## 2024.08.28

- [x] Write plotting scripts for `svr` and `rf` interactions → [[Traditional_ml Plot_svr|dendron://torchcell/experiments.002-dmi-tmi.scripts.traditional_ml-plot_svr]], [[Traditional_ml Plot_random_forest|dendron://torchcell/experiments.002-dmi-tmi.scripts.traditional_ml-plot_random_forest]], [[Traditional_ml Plot_elastic_net|dendron://torchcell/experiments.002-dmi-tmi.scripts.traditional_ml-plot_elastic_net]]
- [x] sync `EN_1e03` → summaries have not been saving. → investigated and submitted issue to wandb. [wandb github issue](https://github.com/wandb/wandb/issues/7227)
- [x] Ran and completed `EN_1e04`
- [x] Running `RF_1e03`, and `RF_1e04` on m1.... → cancelled so we can work on plotting. sweeps to continue. `zhao-group/torchcell_002-dmi-tmi_trad-ml_random-forest_1e03/sweeps/olyxjcv6`, `zhao-group/torchcell_002-dmi-tmi_trad-ml_random-forest_1e04/sweeps/1tus1dbh`
- [x] Continue `RF_1e03` and `RF_1e04`

## 2024.08.29

- [x] Create slides for group presentation
- [x] ![](./assets/drawio/data-duplication-example-scenario.drawio.png)
- [x] Check on db url update. → worked with cPanel to find NCSA resources on campus that could potentially host db. NCSA [Illinois Computes](https://www.ncsa.illinois.edu/about/illinois-computes/).
