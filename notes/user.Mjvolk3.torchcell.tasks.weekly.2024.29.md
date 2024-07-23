---
id: 5jr8kgd7kdw6xthr0epcqy3
title: '29'
desc: ''
updated: 1721590749467
created: 1721067657903
---

## 2024.07.18

- [x] Change [[torchcell.adapters.cell_adapter]] for yaml configuration → [[2024.07.18 - Config Refactor Time Test|dendron://torchcell/torchcell.adapters.cell_adapter#20240718---config-refactor-time-test]] it works! 😅 now making change.

## 2024.07.17

- [x] [[torchcell.datamodels.schema]] updated so we can add data more modularity. This comes down to decoding experiment and experiment reference. → [[2024.07.17 - How we Type|dendron://torchcell/torchcell.datamodels.schema#20240717---how-we-type]]
- [x] Adjust schema for interaction data. [[torchcell.datamodels.schema]]
- [x] Adjust `biocypher/config/torchcell_schema_config.yaml` for interaction data → no changed needed since everything is contained in experiment.
- [x] Since we query datasets we want to have `dataset.name` as property of experiment. Added to [[torchcell.dataset.experiment_dataset]] → added `dataset_name` to `torchcell/biocypher/config/torchcell_schema_config.yaml`
- [x] Check [[torchcell.knowledge_graphs.minimal_kg]] works → had to refactor a bit since not all find replace previously worked... had limited search scope on first refactor.
- [x] Check [[torchcell.knowledge_graphs.minimal_kg]] with docker to see if we bad entries on import. → had to fix `neo4j.conf` read only default setting updated so we can write to db.
- [x] Add `SmfKuzmin2018` dataset for interactions
- [x] Add `DmfKuzmin2018` dataset for interactions
- [x] Add `DmfCostanzo2016` dataset for interactions

## 2024.07.16

- [x] Adjust schema adding in `dataset_name` nodes. → We want to add to experiment data. This is for back tracing.
- [x] Experiment can be linked to a study - We want to query the exact dataset used in this study. Added to yaml and can worry about later when we handle mechanistic aware case study.

## 2024.07.15

- [x] Roll back `numpy` to `1.26.0`, push version, update pypi, rebuild image.
- [x] Host dataset in read only → To adjust permissions we could changed the `chown` in the entrypoint, but then would have to rebuild the image.
- [x] Run a test to see if we can isolate slurm and docker using `cgroup`? → able to solve this with persistent jobs, without directly stress testing docker inherits cgroup and seems to respect resources as long as job persists.
- [x] Fix the import issues - `ExperimentReferenceOf` looks broken. → Fixed issue with parallel experiment reference index rewrite.
- [x] Make sure that dirs are visible with `chown` and `chmod`. Get permissions correct so view all necessary dirs in vscode. → We are not going to worry about this unless needed. Some commented commands added to bottom of build scripts.
- [x] Fix build not recognizing conf → issue was typo in path... we might want different confs for build tests and production builds in `/scratch`
