---
id: rb56ilvfvyt4jktgvi9weq3
title: Hetero_cell_isab_split
desc: ''
updated: 1741676785779
created: 1741127202651
---
## 2025.03.04 - Data with All Reactions

```python
dataset.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  metabolite={
    num_nodes=2534,
    node_ids=[2534],
  },
  reaction={
    num_nodes=4881,
    node_ids=[4881],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 144211],
    num_edges=144211,
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16095],
    num_edges=16095,
  },
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 20960],
    stoichiometry=[20960],
    num_edges=4882,
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5450],
    num_edges=4881,
  }
)
```

```python
dataset[0]
HeteroData(
  gene={
    node_ids=[6605],
    num_nodes=6605,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6605, 0],
    x_pert=[2, 0],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
    fitness=[1],
    fitness_std=[1],
    pert_mask=[6607],
  },
  reaction={
    num_nodes=4881,
    node_ids=[4881],
    pert_mask=[4881],
  },
  metabolite={
    node_ids=[2534],
    num_nodes=2534,
    pert_mask=[2534],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 144102],
    num_edges=144102,
    pert_mask=[144211],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16090],
    num_edges=16090,
    pert_mask=[16095],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5450],
    num_edges=5450,
    pert_mask=[5450],
  },
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 20960],
    stoichiometry=[20960],
    num_edges=4882,
    pert_mask=[20960],
  }
)
```

```python
batch
HeteroDataBatch(
  gene={
    node_ids=[32],
    num_nodes=211352,
    ids_pert=[32],
    cell_graph_idx_pert=[72],
    x=[211352, 0],
    x_batch=[211352],
    x_ptr=[33],
    x_pert=[72, 0],
    x_pert_batch=[72],
    x_pert_ptr=[33],
    gene_interaction=[32],
    gene_interaction_p_value=[32],
    fitness=[32],
    fitness_std=[32],
    pert_mask=[211424],
    batch=[211352],
    ptr=[33],
  },
  reaction={
    num_nodes=156120,
    node_ids=[32],
    pert_mask=[156192],
    batch=[156120],
    ptr=[33],
  },
  metabolite={
    node_ids=[32],
    num_nodes=81079,
    pert_mask=[81088],
    batch=[81079],
    ptr=[33],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 4612016],
    num_edges=[32],
    pert_mask=[4614752],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 514759],
    num_edges=[32],
    pert_mask=[515040],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 174278],
    num_edges=[32],
    pert_mask=[174400],
  },
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 670394],
    stoichiometry=[670394],
    num_edges=[32],
    pert_mask=[670720],
  }
)
```
