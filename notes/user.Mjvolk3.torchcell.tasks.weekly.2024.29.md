---
id: 5jr8kgd7kdw6xthr0epcqy3
title: '29'
desc: ''
updated: 1721263252852
created: 1721067657903
---

## 2024.07.17

- [x] [[torchcell.datamodels.schema]] updated so we can add data more modularity. This comes down to decoding experiment and experiment reference. → [[2024.07.17 - How we Type|dendron://torchcell/torchcell.datamodels.schema#20240717---how-we-type]]
- [x] Adjust schema for interaction data. [[torchcell.datamodels.schema]]
- [x] Adjust `biocypher/config/torchcell_schema_config.yaml` for interaction data → no changed needed since everything is contained in experiment.
- [x] Since we query datasets we want to have `dataset.name` as property of experiment. Added to [[torchcell.dataset.experiment_dataset]] → added `dataset_name` to `torchcell/biocypher/config/torchcell_schema_config.yaml`
- [x] Check [[torchcell.knowledge_graphs.minimal_kg]] works → had to refactor a bit since not all find replace previously worked... had limited search scope on first refactor.
- [ ] Check [[torchcell.knowledge_graphs.minimal_kg]] with docker to see if we bad entries on import.

- [ ] After merge clean up commented code in [[torchcell.adapters.cell_adapter]] and [[torchcell.dataset.experiment_dataset]]

- [ ] Add `Kuzmin2018` dataset for interactions, it is smaller and covers all interactions.

## 2024.07.16

- [x] Adjust schema adding in `dataset_name` nodes. → We want to add to experiment data. This is for back tracing.
- [x] Experiment can be linked to a study - We want to query the exact dataset used in this study. Added to yaml and can worry about later when we handle mechanistic aware case study.

- [ ] Add genes essentiality dataset.
- [ ] Document about gene essentiality source.
- [ ] Add gene essentiality to schema and clearly differentiated from current fitness. Add in transformation to essentiality to growth type phenotype. This should probably be enforced after querying during data selection and deduplication. The rule is something like if we can find some reasonable fixed function for transforming labels we add them. Don't know of a great way of doing this but. Possible we can even add these relations to the Biolink ontology. In theory this could go on indefinitely but I think one layer of abstraction will serve a lot of good at little cost.
- [ ] Add synthetic lethality. Do same as for essentiality.
- [ ] Add expression dataset for mechanistic aware single fitness
- [ ] Add expression from double fitness
- [ ] Add fitness from singles
- [ ] Add fitness from doubles
- [ ] We need a new project documents reproducible procedure on `gh` for restarting slurm, docker, etc.
- [ ] Run container locally with [[torchcell.knowledge_graphs.minimal_kg]] → Had to restart to make sure previous torchcell db was deleted. → struggling with `database/build/build_linux-arm.sh` retrying from build image. → Cannot install CaLM... →
- [ ] Change logo on docs → to do this we need a `torchcell_sphinx_theme`. → cloned, changed all `pyg_spinx_theme` to `torchcell_sphinx_theme`, pushed, trying rebuild.
- [ ] Expand [[paper-outline-02|dendron://torchcell/paper.outline.02]]
- [ ] `ExperimentReferenceOf` looks broken.
- [ ] Make sure ports are getting forwarded correctly and that we can connect to the database over the network. We need to verify that we can connect with the neo4j browser.
- [ ] Try to link docker and slurm with `cgroup`
- [ ] Run build bash script for testing.
- [ ] `gh` Test build under resource constraints.
- [ ] Change logo on docs → to do this we need a `torchcell_sphinx_theme`. → cloned, changed all `pyg_spinx_theme` to `torchcell_sphinx_theme`, pushed, trying rebuild.
- [ ] Remove software update on image entry point

## 2024.07.15

- [x] Roll back `numpy` to `1.26.0`, push version, update pypi, rebuild image.
- [x] Host dataset in read only → To adjust permissions we could changed the `chown` in the entrypoint, but then would have to rebuild the image.
- [x] Run a test to see if we can isolate slurm and docker using `cgroup`? → able to solve this with persistent jobs, without directly stress testing docker inherits cgroup and seems to respect resources as long as job persists.
- [x] Fix the import issues - `ExperimentReferenceOf` looks broken. → Fixed issue with parallel experiment reference index rewrite.
- [x] Make sure that dirs are visible with `chown` and `chmod`. Get permissions correct so view all necessary dirs in vscode. → We are not going to worry about this unless needed. Some commented commands added to bottom of build scripts.
- [x] Fix build not recognizing conf → issue was typo in path... we might want different confs for build tests and production builds in `/scratch`
