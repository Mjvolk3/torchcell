---
id: epexvgtp427jnjyj40bupb8
title: '32'
desc: ''
updated: 1723350449734
created: 1722806044510
---

Changing as of this week, days are going down chronologically. Before it made sense with extremely long notes to have them reversed. [[user.Mjvolk3.torchcell.tasks.deprecated]]. We do it this way in all other notes.

## 2024.08.04

- [x] Check on db build. → failed due to time limit... Great...
- [x] We can optimize this build per edge node type batch sizes. Custom batch sizing in adapters. → we have added some memory reduction mechanisms via yaml

## 2024.08.08

- [x] biocypher-out combine so we can combine partial knowledge graph builds → [[2024.08.08 - How it works|dendron://torchcell/torchcell.database.biocypher_out_combine#20240808---how-it-works]]
- [x] Test that the combine gives the same as all at once. → [[2024.08.08 - Checking that Combine Produces the Same Import Summary|dendron://torchcell/torchcell.knowledge_graphs.dmf_tmi_combine_kg#20240808---checking-that-combine-produces-the-same-import-summary]] → Not working correctly.

## 2024.08.10

- [x] [[2024.08.10 - Troubleshooting Combine to Match Simultaneous Graph Builds|dendron://torchcell/torchcell.knowledge_graphs.dmf_tmi_combine_kg#20240810---troubleshooting-combine-to-match-simultaneous-graph-builds]]
- [x] Get apoc working for run. We need this to run queries with random number permutations. → [apoc release](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/tag/4.4.0.30) → [[Apoc|dendron://torchcell/neo4j.apoc]] → [[apoc.coll.randomItems|dendron://torchcell/neo4j.apoc.coll.randomItems]] → put on GilaHyper and update slurm script.
- [x] Note to track builds [[GilaHyper Builds|dendron://torchcell/database.docker.builds#gilahyper-builds]] → [[GilaHyper 2024-08-08_18-24-33|dendron://torchcell/database.docker.builds#gilahyper-2024-08-08_18-24-33]]
- [x] Start script for `traditional_ml`[[Traditional_ml_dataset|dendron://torchcell/experiments.002-dmi-tmi.scripts.traditional_ml_dataset]]
- [x] Construct `dmi` and `tmi` `kuzmin` kgs

## 2024.08.10

- [ ] Check kgs finished. If not rerun.
- [ ] Rough plan for building interactions dataset.

- [ ] Combine kgs.
- [ ] Import combined kg.
- [ ]
- [ ] Run queries for `1e3`, `1e4`, `1e5`, `1e6`.

***

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
