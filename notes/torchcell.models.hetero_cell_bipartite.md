---
id: qu9llyiiuunnb1pppn0q5uu
title: Hetero_cell_bipartite
desc: ''
updated: 1749764445218
created: 1743187589917
---
## 2025.03.28 - Bipartite Data

```python
dataset
Neo4jCellDataset(1340841)
dataset[0]
HeteroData(
  gene={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6577, 1600],
    x_pert=[2, 1600],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
    fitness=[1],
    fitness_std=[1],
    pert_mask=[6579],
  },
  reaction={
    num_nodes=7122,
    node_ids=[7122],
    pert_mask=[7122],
  },
  metabolite={
    node_ids=[2806],
    num_nodes=2806,
    pert_mask=[2806],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 143715],
    num_edges=143715,
    pert_mask=[143824],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16062],
    num_edges=16062,
    pert_mask=[16067],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5430],
    num_edges=5430,
    pert_mask=[5430],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 26325],
    stoichiometry=[26325],
    num_edges=26325,
  }
)
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
    hyperedge_index=[2, 26325],
    stoichiometry=[26325],
    num_edges=26325,
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5430],
    num_edges=4881,
  }
)
```

## 2025.04.02 - Named Phenotype Labelling

```python
dataset[0]
HeteroData(
  gene={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6577, 1600],
    x_pert=[2, 1600],
    fitness=[1],
    fitness_std=[1],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
    pert_mask=[6579],
  },
  reaction={
    num_nodes=7122,
    node_ids=[7122],
    pert_mask=[7122],
  },
  metabolite={
    node_ids=[2806],
    num_nodes=2806,
    pert_mask=[2806],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 143715],
    num_edges=143715,
    pert_mask=[143824],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16062],
    num_edges=16062,
    pert_mask=[16067],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5430],
    num_edges=5430,
    pert_mask=[5430],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 26325],
    stoichiometry=[26325],
    num_edges=26325,
  }
)
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
    hyperedge_index=[2, 26325],
    stoichiometry=[26325],
    num_edges=26325,
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5430],
    num_edges=4881,
  }
)
```

We have to find specific examples of metabolism perturbation see metabolic gene removal affects graphs. We just delete gpr nodes.

```python
dataset[3]
HeteroData(
  gene={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6577, 1600],
    x_pert=[2, 1600],
    fitness=[1],
    fitness_std=[1],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
    pert_mask=[6579],
  },
  reaction={
    num_nodes=7121,
    node_ids=[7121],
    pert_mask=[7122],
  },
  metabolite={
    node_ids=[2806],
    num_nodes=2806,
    pert_mask=[2806],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 143784],
    num_edges=143784,
    pert_mask=[143824],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16064],
    num_edges=16064,
    pert_mask=[16067],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5429],
    num_edges=5429,
    pert_mask=[5430],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 26320],
    stoichiometry=[26320],
    num_edges=26320,
  }
)
dataset[4]
HeteroData(
  gene={
    node_ids=[6577],
    num_nodes=6577,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6577, 1600],
    x_pert=[2, 1600],
    fitness=[1],
    fitness_std=[1],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
    pert_mask=[6579],
  },
  reaction={
    num_nodes=7121,
    node_ids=[7121],
    pert_mask=[7122],
  },
  metabolite={
    node_ids=[2806],
    num_nodes=2806,
    pert_mask=[2806],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 143786],
    num_edges=143786,
    pert_mask=[143824],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16062],
    num_edges=16062,
    pert_mask=[16067],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5428],
    num_edges=5428,
    pert_mask=[5430],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 26319],
    stoichiometry=[26319],
    num_edges=26319,
  }
)
```
