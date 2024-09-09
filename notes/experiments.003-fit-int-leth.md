---
id: 38h6ocou3r6j9z2qai0x11z
title: 003 Fit Int Leth
desc: ''
updated: 1725926321116
created: 1725919115953
---


## 2024.09.09 - Thinking About Pooling GNNs

We want to start by using a dense pooling method so we can look at gene ontology enrichment. This means we will have to be careful with total parameter size. The best thing to do would probably be to match [[Dcell|dendron://torchcell/torchcell.models.dcell]].

### Parameters in DCell Model

Approximate number of total go nodes.

```python
graph.G_go.number_of_nodes()
5874
```

We reduce total number of nodes for GO model. DCell for 2 dim output for predicting fitness and interaction.

params_dcell: 15,950,032
params_dcell_linear: 100,010
total parameters: 16,050,042

### Message Passing followed by DiffPool

We want to predict both fitness and interactions. Plan to use Codon Frequency size 64 with 2 message passing networks prior to pooling.