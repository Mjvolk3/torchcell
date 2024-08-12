---
id: auzc4lzjlk17i231rflydf9
title: '33'
desc: ''
updated: 1723490071456
created: 1723481104354
---
## 2024.08.12

- [x] Check kgs finished. If not rerun.
- [x] Combine kgs.
- [x] Run import... Had to stop container and restart... Not sure why. Moving one.
- [x] Rough plan for building interactions dataset.

- [x] We caught an issue with reference p_value. It was set to 1.0. This doesn't make any sense. Changed to None.
- [x] Check if old fitness query works. If not we will have to collect ids to reproduce query... even that might not work since the underlying schema has changed... → This query doesn't work because we needed experiment types so we could map decoding for pydantic data validation. Refactoring.
- [x] Refactor [[torchcell.data.neo4j_query_raw]] → #ramble we call it reference instead of control because it contains genome, which isn't a description of a control experiment, although it does describe the control state. Controls are often thought of as the experiment phenotype and we want to avoid this confusion with a more expansive description. → Refactor completed, the small tests in main work. We can now run the queries any of the experiments from the schema and not just fitness by use a type map in [[torchcell.datamodels.schema]]. We also adjust the processing so we have a write `db` for processing and subsequent runs use a read only `db`. [[2024.08.12 - Neo4jQueryRaw is Immutable|dendron://torchcell/torchcell.data.neo4j_query_raw#20240812---neo4jqueryraw-is-immutable]]

- [ ] Run query. We want all `dmi_kuzmin_2018`, add `tmi_kuzmin_2018`, and then partials of `dmi_costanzo_2016`.
- [ ] Run queries for `1e3`, `1e4`, `1e5`, `1e6`.
- [ ] Build datasets with embeddings without any plotting.
- [ ] Plots on data distributions.

- [ ]

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
