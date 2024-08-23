---
id: 65w5kdtce7q047b7ollalhq
title: '34'
desc: ''
updated: 1724353836135
created: 1724101120841
---
## 2024.08.19 

- [ ] Solve speed issue with loader by using a cpu loader. Current issue is that the iteration speeds are super low because we are not using multiprocessing on loading datasets for building `train`, `val`, `test` → 
- [ ] Bring in kuzmin2020 dataset
- [ ] Plots on data distributions.

## 2024.08.22

- [ ] Check if `Neo4jCell` can be pickled without deleting vars.
- [ ]


***

- [ ] Zendron on `zotero_out`
- [ ] Add in transformation to essentiality to growth type phenotype. This should probably be enforced after querying during data selection and deduplication. The rule is something like if we can find some reasonable fixed function for transforming labels we add them. Don't know of a great way of doing this but. Possible we can even add these relations to the Biolink ontology. In theory this could go on indefinitely but I think one layer of abstraction will serve a lot of good at little cost.
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
