---
id: hvoq1nnkwb2ng0tytnndn8v
title: 094310
desc: ''
updated: 1710859393044
created: 1710859393044
---
```python
dataset.cell_graph
HeteroData(
  gene={
    num_nodes=6579,
    node_ids=[6579],
    x=[6579, 1536],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 139102],
    num_edges=139102,
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 9498],
    num_edges=9498,
  }
)
```

```python
dataset[0]
HeteroData(
  gene={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6577, 1536],
    x_pert=[2, 1536],
    graph_level='global',
    label='dmf',
    label_error='dmf_std',
    label_value=0.1749,
    label_value_std=0.0,
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 139052],
    num_edges=139052,
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 9492],
    num_edges=9492,
  }
)
```
