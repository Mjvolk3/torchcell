---
id: qrpuop083pghmfiqe6va4y7
title: torchcell.tasks.future
desc: ''
updated: 1697580479754
created: 1675887826601
---
## Future

- [ ] Notes that represent a python file should have the path to the python file in the frontmatter.
- [ ] Convert `gene_set` to `SortedSet`
- [ ] Workspace utils from note open related python file
- [ ] Consider `BaseEmbeddingDataset` returning a graph instead.
- [ ] Train DCell model
- [ ] Separate out CI more, specifically style
- [ ] Add args so this goes into data dir `dbfn="data.db",`
- [ ] `torch_geometric/deprecation.py:22: UserWarning: 'data.DataLoader' is deprecated, use 'loader.DataLoader' instead`
- [ ] Add tiling window functions for nucleotide transformer
- [ ] Speed up filtering in `cell.py`, this works well on [[2023.09.07|dendron://torchcell/user.Mjvolk3.torchcell.tasks#20230907]] `M1`, but very slow on `Delta`
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

## Far Future

- [ ] Download phenotypes from [SGD phenotypes](http://sgd-archive.yeastgenome.org/curation/literature/)
- [ ] Consider batching i.e. lists of list of sequences. Can be done on GPU. Note that this causes things to crash on local.
- [ ] Move over analysis of the mutant datasets from `Gene_Graph`
- [ ] Language model embedding of gene description. Language model can be pretrained on all of the relevant yeast literature. Then embeddings can be generated for the descriptions. There is probably more text we can pull. With @Heng-Ji we could probably get an ego graph for each gene. Would be nice if could match it to some sort of experimental ontology. If you find these features are important we could try to do additional computation on top of the ego graph from literature. As of now we could use the literature list that his hosted by SGD. These paperms could be scraped, plus there supplementary pdfs.

```python
genome["YDR210W"].attributes['display'][0]
'Predicted tail-anchored plasma membrane protein
```
