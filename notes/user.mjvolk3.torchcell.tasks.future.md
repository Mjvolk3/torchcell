---
id: qrpuop083pghmfiqe6va4y7
title: torchcell.tasks.future
desc: ''
updated: 1742343933578
created: 1675887826601
---
## Future

- [ ] graph, bipartite, hypergraph is detectable by definition of node. We could just automatically detect this.
- [ ] See if we can improve the convergence on `pma`. → On training we have the issue that it likes to predict just a few values, in other words the distribution is spiky.

***

- [ ] Fix pin memory not working due to some data being sent to device prior to the pinning.
- [ ] Fix slow subgraph representation.

- [ ] Add after we get run going on GPU [[torchcell.viz.visual_regression]]

- [ ] Edit to get most up to date formula of the problem. [[Isomorphic_cell|dendron://torchcell/torchcell.models.isomorphic_cell]]

***
**Node Embeddings for Whole Genome**

- [ ] Delay to feature transfer
- [ ] Find and replace str for moving node embeddings.
- [ ] Move all node embeddings `M1`.
- [ ] Delete Node embeddings on `Delta`.
- [ ] Transfer Node embeddings to `Delta`.
- [ ] Remove node embeddings on `GH` so when we get machine on return things will break until we transfer node embeddings back.

- [ ] Wait on this... plots for enrichment of all genes. Contains any of all graphs.
- [ ] Make sure y distributions look like they y from other datasets.
- [ ] Histogram of genes usage in subset... we have these plots somewhere.
- [ ] Evaluate
500,000 (5e5) - cannot do with current db we have
500,000 (5e5)

- [ ] Want small datasets with most possible perturbations. Check network overlap?

***

### Writing

- [ ] [[Outline 02|dendron://torchcell/paper.outline.02]] - Move on to outline 3 too much has changed. [[Outline 03|dendron://torchcell/paper.outline.03]]

- [ ] Start Draft pipeline. Bring in thesis latex template.

***

- [ ] Test edge attention on hypergraph conv
- [ ] Update `cell_latent_perturbation`
- [ ] `cell_latent_perturbation` remove stoichiometry for reaction aggregation
- [ ] `cell_latent_perturbation` unify embeddings across graphs
- [ ] `cell_latent_perturbation` use `ISAB` set transformer for `intact_whole`
- [ ] unified model

- [ ] Memory issue with regression to classification scripts. We still have issue of processing memory accumulation. Unsure where it is coming from. Will only need to be solved if we use these losses.
- [ ] Implement intact and pert phenotype processor.
- [ ] Synthesize Results in report. Discussion on consideration of use alternative methods like mse plus a divergence measure.
- [ ] Run metabolism label split regression run
- [ ] Information Diff., WL Kernel

## Notes on Metabolism

- Can get Gibbs Free Energy of reaction from [MetaCyc](https://biocyc.org/reaction?orgid=META&id=D-LACTATE-DEHYDROGENASE-CYTOCHROME-RXN)
- To preserve sign information in [[Met_hypergraph_conv|dendron://torchcell/torchcell.nn.stoichiometric_hypergraph_conv]] we should use activations that can handle negative input like leaky relu, elu, or tanh.

## Notes Related to Dango

Breakout into specific notes on Dango.

- [ ] Verify

> Pearson correlation between the trigenic interaction scores of two individual replicates is around 0.59, which is much lower than the Pearson correlation between the digenic interaction score of two replicates from the same data source (0.88). ([Zhang et al., 2020, p. 3](zotero://select/library/items/PJFDVT8Y)) ([pdf](zotero://open-pdf/library/items/AFBC5E89?page=3&annotation=D8D949VF))

- [ ] Plot P-Values of current dataset to compare to predicted interactions. Can do for both digenic and trigenic interactions. Do this over queried datasets.
- [ ] What is purpose of the pretraining portion? Why not just take embeddings and put into this hypergraph embedding portion?

***

[[04|dendron://torchcell/user.Mjvolk3.torchcell.tasks.weekly.2025.04]]

- [ ] Update combine to add a `README.md` which can serve as a trace to combined data.
- [ ] Combined datasets and update readonly db.
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
- [ ] Check ontology vs SGD ontology GAF.
- [ ] Compute flops of different networks. There are python libraries to do this for `nn.Module`
- [ ] From datasets I think it would be nice to return the data objects, but then adapters would have to be fixed. We opted not to do this is originally because it made multiprocessing easier, but I think we can use the deserialization in the adapter if we write the model and just make `transform_item` transform into dict, then it would be much more like a dump method. Should be done after pipeline completion.
- [ ] Use omega conf to select the functions that get called automatically for `get_nodes` and `get_edges`.
- [ ] Formal docker import test with subprocess docker. Create unique containers for each dataset. Output the and evaluate the report file.
- [ ] [[tests.torchcell.adapters.test_costanzo2016_adapter]] and [[tests.torchcell.adapters.test_kuzmin2018_adapter]] create extra longs when using multiprocessing that should be deleted.
- [ ] Test for datasets that read out the docker neo4j bulk import report.
- [ ] Automate the database folder creation given whichever architecture is specified, or detected.
- [ ] Publish package for Junyu and Le to use
- [ ] Get rid of the `preprocess_config.json
- [ ] Change `reference_environment` to `environment`
- [ ] #pr.biocypher.import_call_file_prefix, path mapping `import_call_file_prefix`
- [ ] #pr.biocypher, message @Sebastian-Lobentanzer about collectri pr accept on readme
- [ ] #pr.biocypher minimal example for docker `None` import. → I have a suspicion that the that the docker build importing `None` doesn't work with the biocypher provided docker compose because of the mismatched neo4j version. Unsure..
- [ ] #pr.biocypher update tutorials with Neo4j Bloom images.
- [ ] Notify @Sebastian-Lobentanzer about #pr.biocypher [Collectri ReadMe Update](https://github.com/biocypher/collectri/pull/1).

- [ ] Front matter fix
`.py` on github not showing up...

```python
# torchcell/knowledge_graphs/create_scerevisiae_kg_small
# [[torchcell.knowledge_graphs.create_scerevisiae_kg_small]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/create_scerevisiae_kg_small
# Test file: tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg_small.py
```

- [ ] For simplicity we should get rid of `preprocess_dir` and put everything into `process_dir`
- [ ] Embeddings should be union over `gene_sets`
- [ ] We have this strange issue that whatever is run from the terminal in docker ends up getting prepended to the `neo4j.conf` file... Not sure how this makes any sense. [[2024.03.21 - Terminal Commands Copied into neo4j.conf File|dendron://torchcell/neo4j.conf#20240321---terminal-commands-copied-into-neo4jconf-file]]
- [ ] Add dataset to experiment and reference for index subsetting by dataset. This probably makes most sense for reducing `dmf` count. We would like the coherence for keeping as much `dmf` data from `Kuzmin` as possible.
- [ ] [[2024.04.04 - MODEL_TO_WINDOW changed to some configuration|dendron://torchcell/torchcell.datasets.embedding#20240404---model_to_window-changed-to-some-configuration]]
- [ ] in `IndexSplit` changes `indices` to `index`...

## Far Future

- [ ] We have multiple `.bashrc` that are competing with apptainer and the base env on delta... fix this
- [ ] The thought on duplicates is to provide a deduplicator class that handles the duplicates. This depends on the details of how the modeler chooses to model the domain so we should just design an interface for doing so. For now I am hardcoding this in. → Writing a deduplicator class.
- [ ] Deep Set models parameterize by add and mean for `scatter_add` and `scatter_mean`
- [ ] Consider using `nn.Embedding` for categoricals
- [ ] Write Cron Job for Db build locally
- [ ] CI/CD for docker image builds queued on changes in Dockerfile or version of `torchcell` lib.
- [ ] Change `wandb.tags` to `Tags`
- [ ] Setup `test/data`
- [ ] Notes that represent a python file should have the path to the python file in the frontmatter.
- [ ] Convert `gene_set` to `SortedSet`
- [ ] Workspace utils from note open related python file
- [ ] Consider `BaseEmbeddingDataset` returning a graph instead.
- [ ] Train DCell model
- [ ] Separate out CI more, specifically style
- [ ] Add args so this goes into data dir `dbfn="data.db",`
- [ ] `torch_geometric/deprecation.py:22: UserWarning: 'data.DataLoader' is deprecated, use 'loader.DataLoader' instead`
- [ ] Add tiling window functions for nucleotide transformer
- [ ] Speed up filtering in `cell.py`, this works well on @Mjvolk3.torchcell.tasks.deprecated.2024.06.18 `M1`, but very slow on `Delta`
- [ ] Consider making an `smf` dataset that comes from the `dmf` data.
- [ ] visualize the dependency of the library (10 min)
- [ ] We need reason to believe that using llm should work. Collect `1e5` dataset, `add`, `mean`, vectors of missing data, umap visualize, with dmf overlay
- [ ] Do same `umap` for `smf` alone.
- [ ] If both of `smf` and `dmf` umap look to work, do a combined umap, with `smf` as a different shape.
- [ ] Gene ontology for `DCell`
- [ ] `DCell` model
- [ ] Write `DCell` network as perturbation to GO graph
- [ ] WT difference for loss function... thinking dataset should have a reference object at highest level.
- [ ] WL-Lehman for fitness prediction
- [ ] Add in gene essentiality dataset `smf`
- [ ] Add in synthetic lethality dataset `dmf` [synthetic lethality db](https://synlethdb.sist.shanghaitech.edu.cn/v2/#/) this doesn't look like it has media conditions.
- [ ] Rewrite single cell fitness for `lmdb`
- [ ] Work on merge single cell fitness data
- [ ] Add triple mutant fitness dataset `tmf`
- [ ] Add gene expression for `smf` data
- [ ] Add gene expression data for `dmf` data
- [ ] Add morphology dataset
- [ ] Add plotting functionality on genomes
- [ ] I am thinking that `CellDataset` is going to be so complex that we will need some sort of configuration
- [ ] When I do joins of data I want to know what types of data were excluded and which were included. I think that there operations need to be part of something like `Cell.join`
- [ ] Sort out inconsistencies with embedding datasets. Try to unify with `abc`.
- [ ] Infer gene ontology from DNA embeddings. First find the portion of the ontology that is occupied by proteins, enzymes, etc.
- [ ] Download phenotypes from [SGD phenotypes](http://sgd-archive.yeastgenome.org/curation/literature/)
- [ ] Consider batching i.e. lists of list of sequences. Can be done on GPU. Note that this causes things to crash on local.
- [ ] Move over analysis of the mutant datasets from `Gene_Graph`
- [ ] Language model embedding of gene description. Language model can be pretrained on all of the relevant yeast literature. Then embeddings can be generated for the descriptions. There is probably more text we can pull. With @Heng-Ji we could probably get an ego graph for each gene. Would be nice if could match it to some sort of experimental ontology. If you find these features are important we could try to do additional computation on top of the ego graph from literature. As of now we could use the literature list that his hosted by SGD. These paperms could be scraped, plus there supplementary pdfs.

```python
genome["YDR210W"].attributes['display'][0]
'Predicted tail-anchored plasma membrane protein
```

- [ ] First `Ollama` application to write documentation, write tests. Both should come with summaries. Automatically run tests to inspect which tests fail. Test if new dataset code is adherent to schema or can be made adherent to schema. We should also generate schema expansion recommendations. This part is hard, and really needs a vast amount of knowledge over different experimental methods if we want it to scale to 1000s of studies. This will build our understanding of ontologizing in the domain along with predictions over the ontology. This should be evidence enough for funding parallelized pilot scale reactors. 100 L to one 1000 L reactor. Once the we can reliably use the 1000 L reactor to produce a product at profit, we should be able to achieve a second round of funding for replicating the process. So we can penetrate the market.
