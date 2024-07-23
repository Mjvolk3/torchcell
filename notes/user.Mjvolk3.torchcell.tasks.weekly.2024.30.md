---
id: 2udnurpk5ce6e78o55plv93
title: '30'
desc: ''
updated: 1721770377641
created: 1721590753227
---

## 2024.07.23

- [x] Add pubmed id to dataset in schema.
- [x] After merge clean up commented code in [[torchcell.adapters.cell_adapter]] and [[torchcell.dataset.experiment_dataset]]
- [ ] Add gene essentiality dataset.

- [ ] Add synthetic lethality. Do same as for essentiality.

- [ ] Add gene essentiality to schema and clearly differentiated from current fitness. Add in transformation to essentiality to growth type phenotype. This should probably be enforced after querying during data selection and deduplication. The rule is something like if we can find some reasonable fixed function for transforming labels we add them. Don't know of a great way of doing this but. Possible we can even add these relations to the Biolink ontology. In theory this could go on indefinitely but I think one layer of abstraction will serve a lot of good at little cost.

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
- [ ] dataset registry not working again because circular import

## 2024.07.21

- [x] After merge delete feature branch. Drop all stashes
- [x] Add genes essentiality dataset. → requires fix on genome looks like SGD changed path names. → fixed url for [[torchcell.sequence.genome.scerevisiae.s288c]] but this doesn't contain gene data needed for essentiality. → in the works.
