---
id: goislb7hg0rffoj4hhlzawl
title: '27'
desc: ''
updated: 1720728200298
created: 1719839145091
---

## 2024.07.07

- [ ] Get https working with certificates setup for public and private keys
- [ ] Make sure that dirs are visible with `chown` and `chmod`
- [ ] Biochatter ChatGSE compose?

## 2024.07.06

- [x] Check in on container that was running over night. → Has been up for 19 hours uninterrupted. → Only issues is I have no idea how and why it is still up.
- [x] Run basic query and see if it stays up... check at midnight. → still up.
- [x] #ramble It has become clear that the initialization from scratch is partially manual process. At least the end is so clunky that it it's is unclear if it will be robust enough to automate. I think we should view this script one shot creation + some manual setup. Future should be just about generating `biocypher-out` for new knowledge graph. Unless we can stop database from outside of cypher shell. → build can be separated into two parts. [[2024.07.06 - Building on GilaHyper Workstation|dendron://torchcell/database.docker.build.overview#20240706---building-on-gilahyper-workstation]]

- [x] Add setup script to slurm script. → had to revert, getting weird error with setup and changing permissions. → Still cannot https working, applying s

- [ ] Get permissions correct so view all necessary dirs in vscode.
- [ ] See if we can make ports https protocol.

- [ ] Run a test to see if we can isolate slurm and docker using `cgroup`?
- [ ] Fix the import issues - `ExperimentReferenceOf` looks broken.

- [ ] Adjust schema adding in `dataset_name` nodes.

- [ ] Adjust datasets accounting for `dataset_name` nodes.

- [ ] Add `Kuzmin2018` dataset for interactions, it is smaller and covers all interactions.
- [ ] Adjust schema for interaction data.
- [ ] Add interactions to adapter.
- [ ] Add genes essentiality dataset.
- [ ] Document about gene essentiality source.
- [ ] Add gene essentiality to schema and clearly differentiated from current fitness. Add in transformation to essentiality to growth type phenotype. This should probably be enforced after querying during data selection and deduplication. The rule is something like if we can find some reasonable fixed function for transforming labels we add them. Don't know of a great way of doing this but. Possible we can even add these relations to the Biolink ontology. In theory this could go on indefinitely but I think one layer of abstraction will serve a lot of good at little cost.

- [ ] Add synthetic lethality. Do same as for essentiality.
- [ ] Add expression dataset for mechanistic aware single fitness
- [ ] Add expression from double fitness
- [ ] Add fitness from singles
- [ ] Add fitness from doubles
- [ ] Experiment can be linked to a study - We want to query the exact dataset used in this study.

- [ ] We need a new project documents reproducible procedure on `gh` for restarting slurm, docker, etc.

- [ ] Run container locally with [[torchcell.knowledge_graphs.minimal_kg]] → Had to restart to make sure previous torchcell db was deleted. → struggling with `database/build/build_linux-arm.sh` retrying from build image. → Cannot install CaLM... →

- [ ] Change logo on docs → to do this we need a `torchcell_sphinx_theme`. → cloned, changed all `pyg_spinx_theme` to `torchcell_sphinx_theme`, pushed, trying rebuild.

## 2024.07.05

- [ ] Expand [[paper-outline-02|dendron://torchcell/paper.outline.02]]

- [x] [[2024.07.06 - Create Database|dendron://torchcell/cypher-shell#20240706---create-database]]
- [x] [[2024.07.06 - Node Count and Edge Count|dendron://torchcell/cypher-shell#20240706---node-count-and-edge-count]]
- [x] Run container locally with [[torchcell.knowledge_graphs.minimal_kg]] → Had to restart to make sure previous torchcell db was deleted. → struggling with `database/build/build_linux-arm.sh` retrying from build image. → Cannot install CaLM... → moved tomorrow
- [x] Start database. → Can get it to start but stops shortly there after. `databases` and `transactions` must be owned by neo4j. → We have to create the `torchcell` database from the cypher-shell, then restart the container. → The database is available <http://gilahyper.zapto.org:7474/>. We have an issue of docker container randomly closing. I think it might have to do with the neo4j threads not matching cpus given in run.
- [x] Try rerun with 10 cpus. → `dbms.threads.worker_count=10` → failed
- [x] Check to make sure the container has permissions over all dirs. This could cause issue. → seems that permissions within database aren't an issue. → Maybe it is a network error that creates disruption. → Database has stayed up 12 minutes, and has been queried from remote. Leaving to see if it stays up overnight. → Overnight things worked... I have to guess it was some networks errors... but this remains a little unsatsifying with no error messaging.
- [x] While getting coffee make sure that we can connect to the neo4j browser. → it works remotely.

- [ ] `ExperimentReferenceOf` looks broken.

- [ ] Make sure ports are getting forwarded correctly and that we can connect to the database over the network. We need to verify that we can connect with the neo4j browser.
- [ ] Try to link docker and slurm with `cgroup`
- [ ] Run build bash script for testing.
- [ ] `gh` Test build under resource constraints.
- [ ] Change logo on docs → to do this we need a `torchcell_sphinx_theme`. → cloned, changed all `pyg_spinx_theme` to `torchcell_sphinx_theme`, pushed, trying rebuild.

## 2024.07.04

- [x] Get docker going again on `gh` run test sbatch → need to bump version for small.
- [x] Bump version → fix. colon? → looks like we do need colon.
- [x] Run small kg → Verify that things update on push to main.
- [x] Correct setup script so we can start fresh from `/scratch`. We deleted `wandb-experiments` but they are saved online. → I think setting `wandb-experiments` will automatically create them when they are necessary.
- [x] Make sure that permissions are set such that we can still view all of the files in `/database`
