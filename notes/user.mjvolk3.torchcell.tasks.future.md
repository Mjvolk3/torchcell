---
id: qrpuop083pghmfiqe6va4y7
title: torchcell.tasks.future
desc: ''
updated: 1693252262030
created: 1675887826601
---
## Future

- [ ] Workspace utils from note open related python file
- [ ] Consider `BaseEmbeddingDataset` returning a graph instead.
- [ ] Train DCell model
- [ ] Separate out CI more, specifically style
- [ ] Add args so this goes into data dir `dbfn="data.db",`
- [ ] `torch_geometric/deprecation.py:22: UserWarning: 'data.DataLoader' is deprecated, use 'loader.DataLoader' instead`
- [ ] Add tiling window functions for nucleotide transformer

## Far Future

- [ ] Consider batching i.e. lists of list of sequences. Can be done on GPU. Note that this causes things to crash on local.
- [ ] Move over analysis of the mutant datasets from `Gene_Graph`
- [ ] Language model embedding of gene description. Language model can be pretrained on all of the relevant yeast literature. Then embeddings can be generated for the descriptions. There is probably more text we can pull. With @Heng-Ji we could probably get an ego graph for each gene. Would be nice if could match it to some sort of experimental ontology. If you find these features are important we could try to do additional computation on top of the ego graph from literature. As of now we could use the literature list that his hosted by SGD. These paperms could be scraped, plus there supplementary pdfs.

```python
genome["YDR210W"].attributes['display'][0]
'Predicted tail-anchored plasma membrane protein'
```
