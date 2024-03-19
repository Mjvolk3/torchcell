---
id: 4km5bfhgot4d6ocr65hp62w
title: '114625'
desc: ''
updated: 1710728958819
created: 1710693988695
---

Using the safe compose between the `G_physical` and the `G_regulatory`

self.cell_graph
Data(edge_index=[2, 286416], num_nodes=6707, ids=[6707])

***

```python
graph.G_regulatory.number_of_edges()
>>> 9753
```

```python
graph.G_physical.number_of_edges()
>>> 139463
```

We have to force the G_physical do directed since the regulatory network is directed. so we get 2x the number of edges `278,926 + 9753 = 288,679 != 286416`... Not sure how this could be.

***

We want the subsetted_graph to be of this form.

Here is `cell_graph`

```python
HeteroData(
  (physical, interacts, physical)=Data(edge_index=[2, 276347], num_nodes=5721),
  (regulatory, interacts, regulatory)=Data(edge_index=[2, 9498], num_nodes=3632),
  (base, interacts, base)=Data(edge_index=[2, 0], num_nodes=6579),
  physical={
    node_ids=[6579],
    num_nodes=6579,
  },
  regulatory={
    node_ids=[6579],
    num_nodes=6579,
  },
  base={
    node_ids=[6579],
    num_nodes=6579,
  }
)
```

Here is subsetted_graph example output

```python
HeteroData(
  (physical, interacts, physical)=Data(edge_index=[2, 276320], num_nodes=5719),
  (regulatory, interacts, regulatory)=Data(edge_index=[2, 9492], num_nodes=3631),
  (base, interacts, base)=Data(edge_index=[2, 0], num_nodes=6577),
  physical={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    x_pert_idx=[2],
  },
  regulatory={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    x_pert_idx=[2],
  },
  base={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    x_pert_idx=[2],
  }
)
```

Notice how the num_nodes change and num_edges

***

```python
return processed_graph # at debugger breakpoint
processed_graph
HeteroData(
  physical={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    x_pert_idx=[2],
  },
  regulatory={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    x_pert_idx=[2],
  },
  base={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    x_pert_idx=[2],
  }
)
cell_graph.edge_types
[]
cell_graph.node_types
['physical', 'regulatory', 'base']
```

***

```python
return processed_graph # at debugger breakpoint
processed_graph
HeteroData(
  gene={
    x=[]
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    x_pert_idx=[2],
  },
  regulatory={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    x_pert_idx=[2],
  },
  base={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    x_pert_idx=[2],
  }
)
cell_graph.edge_types
[]
cell_graph.node_types
['physical', 'regulatory', 'base']
```

***

```python
genome.gene_set
GeneSet(size=6579, items=['YAL001C', 'YAL002W', 'YAL003W']...)
fudt_3prime_dataset
FungalUpDownTransformerDataset(6579)
genome.gene_set[0]
'YAL001C'
fudt_3prime_dataset
FungalUpDownTransformerDataset(6579)
fudt_3prime_dataset[0]
Data(
  id='YAL001C',
  dna_windows={ species_downstream=id='YAL001C' chromosome=1 strand='-' start=147594 end=151166 seq='TAATGAAATGAGGTGTATAAATTTTACTTTTATGTAACCAAAGTTGTATTAAATATTTAGAAATGTTATACTATTTTTGGGTTAGATTCCGTCTGGCAAATTAAACAAGAATATTCATCGGGTTTCTGGGCCAAGTTTTCGAGGCAAGTCTGGTGAAAGCCATGGTGACATTTGAATATGACAAGGGGAGTTTTGAGATCTACACTAATCATATCTTGGTGGCGCTGTACATTTTCCCAAGCTAGAAAAAGTAATGGGTCCAGACCAGCTCCCCATATTTTTTTCCCGCAGATTTCGCAG' start_window=147296 end_window=147596 },
  embeddings={ species_downstream=[1, 768] }
)
fudt_5prime_dataset
FungalUpDownTransformerDataset(6579)
fudt_5prime_dataset[0]
Data(
  id='YAL001C',
  dna_windows={ species_downstream=id='YAL001C' chromosome=1 strand='-' start=147594 end=151166 seq='TAATGAAATGAGGTGTATAAATTTTACTTTTATGTAACCAAAGTTGTATTAAATATTTAGAAATGTTATACTATTTTTGGGTTAGATTCCGTCTGGCAAATTAAACAAGAATATTCATCGGGTTTCTGGGCCAAGTTTTCGAGGCAAGTCTGGTGAAAGCCATGGTGACATTTGAATATGACAAGGGGAGTTTTGAGATCTACACTAATCATATCTTGGTGGCGCTGTACATTTTCCCAAGCTAGAAAAAGTAATGGGTCCAGACCAGCTCCCCATATTTTTTTCCCGCAGATTTCGCAG' start_window=147296 end_window=147596 },
  embeddings={ species_downstream=[1, 768] }
)
```
***

`ids_pert` list of genes deleted.
`cell_graph_pert_idx` - the corresponding index on cell graph that was removed.
`x_pert` - also please add this which would be the embeddings genes removed