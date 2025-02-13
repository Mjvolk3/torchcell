---
id: fv03p7fr93q6yfck4hzfg9y
title: Isomorphic_cell_attentional
desc: ''
updated: 1739422637980
created: 1738200326629
---
## 2025.02.12 - Data Masking

```python
cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 64],
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
)```

```python
dataset[0]
HeteroData(
  gene={
    node_ids=[6605],
    num_nodes=6605,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6605, 64],
    x_pert=[2, 64],
    fitness=[1],
    fitness_std=[1],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
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
    node_ids=[4],
    num_nodes=26417,
    ids_pert=[4],
    cell_graph_idx_pert=[11],
    x=[26417, 64],
    x_batch=[26417],
    x_ptr=[5],
    x_pert=[11, 64],
    x_pert_batch=[11],
    x_pert_ptr=[5],
    gene_interaction=[4],
    gene_interaction_p_value=[4],
    fitness=[4],
    fitness_std=[4],
    pert_mask=[26428],
    batch=[26417],
    ptr=[5],
  },
  reaction={
    num_nodes=19523,
    node_ids=[4],
    pert_mask=[19524],
    batch=[19523],
    ptr=[5],
  },
  metabolite={
    node_ids=[4],
    num_nodes=10136,
    pert_mask=[10136],
    batch=[10136],
    ptr=[5],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 576619],
    num_edges=[4],
    pert_mask=[576844],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 64363],
    num_edges=[4],
    pert_mask=[64380],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 21799],
    num_edges=[4],
    pert_mask=[21800],
  },
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 83835],
    stoichiometry=[83835],
    num_edges=[4],
    pert_mask=[83840],
  }
)
```
