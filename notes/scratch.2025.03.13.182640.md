---
id: fw083qnozdssexbslniv384
title: '182640'
desc: ''
updated: 1741908453057
created: 1741908404523
---
Data

```python
dataset.cell_graph
HeteroData(
  gene={
    num_nodes=6579,
    node_ids=[6579],
    x=[6579, 1600],
  },
  metabolite={
    num_nodes=2806,
    node_ids=[2806],
  },
  reaction={
    num_nodes=7122,
    node_ids=[7122],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 143824],
    num_edges=143824,
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16067],
    num_edges=16067,
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 13580],
    stoichiometry=[13580],
    edge_type=[13580],
    num_edges=13580,
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5430],
    num_edges=4881,
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
    x=[6579, 1600],
    x_pert=[2, 1600],
    fitness=[1],
    fitness_std=[1],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
    pert_mask=[6579],
    fitness_original=[1],
    gene_interaction_original=[1],
    mask=[6579],
  },
  reaction={
    num_nodes=7122,
    node_ids=[7122],
    pert_mask=[7122],
    mask=[7122],
  },
  metabolite={
    node_ids=[2806],
    num_nodes=2806,
    pert_mask=[2806],
    mask=[2806],
  },
  (gene, physical_interaction, gene)={
    num_edges=143715,
    pert_mask=[143824],
    adj=[6579, 6579],
  },
  (gene, regulatory_interaction, gene)={
    num_edges=16062,
    pert_mask=[16067],
    adj=[6579, 6579],
  },
  (gene, gpr, reaction)={
    num_edges=5430,
    pert_mask=[5430],
    inc_matrix=[6579, 7122],
  },
  (reaction, rmr, metabolite)={
    stoichiometry=[13580],
    edge_type=[13580],
    num_edges=13580,
    pert_mask=[13580],
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
    inc_matrix=[7122, 2806],
    weighted_inc_matrix=[7122, 2806],
    edge_type_matrix=[7122, 2806],
  }
)
```
