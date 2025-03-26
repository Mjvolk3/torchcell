---
id: p64k8x1u54badkcybt1vcpn
title: Neo4j_cell
desc: ''
updated: 1741759964473
created: 1727139151784
---
## 2024.10.14 - Idea for Supporting Different Base Graphs

```python
def get_init_graphs(self, raw_db, genome):
    # Setting priority
    if genome is None:
        cell_graph = create_graph_from_gene_set(raw_db.gene_set)
    elif genome:
        cell_graph = create_graph_from_gene_set(genome.gene_set)
    return cell_graph
```

## 2025.03.11 - Metabolism Hypergraph and Metabolism Bipartite

We can view the metabolism as a hypergraph and it's associated bipartite.

GPR edges don't require edge types since they all represent the same relationship (gene association with reaction), whereas RMR edges need edge types to distinguish between reactants (0) and products (1), which is essential for modeling metabolic pathway directionality and identifying isolated products during perturbations.

**Metabolism Hypergraph**

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
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 26325],
    stoichiometry=[26325],
    num_edges=7123,
    reaction_to_genes=dict(len=7122),
    reaction_to_genes_indices=dict(len=7122),
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5430],
    num_edges=4881,
  }
)
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
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 26320],
    stoichiometry=[26320],
    num_edges=7122,
    pert_mask=[26325],
  }
)
```

**Metabolism Bipartite**

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
    edge_index=[2, 13580],
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
    edge_index=[2, 13577],
    stoichiometry=[13577],
    edge_type=[13577],
    num_edges=13577,
    pert_mask=[13580],
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  }
)
```

batch

```python
batch
HeteroDataBatch(
  gene={
    node_ids=[2],
    num_nodes=13211,
    ids_pert=[2],
    cell_graph_idx_pert=[3],
    x=[13211, 0],
    x_batch=[13211],
    x_ptr=[3],
    x_pert=[3, 0],
    x_pert_batch=[3],
    x_pert_ptr=[3],
    fitness=[2],
    fitness_std=[2],
    gene_interaction=[2],
    gene_interaction_p_value=[2],
    pert_mask=[13214],
    batch=[13211],
    ptr=[3],
  },
  reaction={
    num_nodes=14242,
    node_ids=[2],
    pert_mask=[14244],
    batch=[14242],
    ptr=[3],
  },
  metabolite={
    node_ids=[2],
    num_nodes=5609,
    pert_mask=[5612],
    batch=[5609],
    ptr=[3],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 288296],
    num_edges=[2],
    pert_mask=[288422],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 32196],
    num_edges=[2],
    pert_mask=[32202],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 10898],
    num_edges=[2],
    pert_mask=[10900],
  },
  (reaction, rmr, metabolite)={
    edge_index=[2, 27156],
    stoichiometry=[27156],
    edge_type=[27156],
    num_edges=[2],
    pert_mask=[27160],
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  }
)
```