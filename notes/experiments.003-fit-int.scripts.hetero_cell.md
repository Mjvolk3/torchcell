---
id: rnkzfrrchb9iqb8jpstv0ek
title: Hetero_cell
desc: ''
updated: 1741661416769
created: 1740603188397
---
## 2025.03.10 - Including Reactions that Don't have Gene Associations

We are now including reactions that don't have genes. This is done in [[yeast_GEM|dendron://torchcell/torchcell.metabolism.yeast_GEM]]. Before today we were not doing this. The logic of doing this is that it should increase the level of connectivity of metabolism. This will also help give a more principled replication of metabolism which will in turn help when later considering small molecule interactions.

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
```

Before `dataset.cell_graph` looked like this.

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

Since we are still using the hypergraph representation here we see that we have dramatically increased the number of reactions `7,122 - 4,822 = 2,300`. If we consider the sum of `num_metabolites`, `num_genes`, and `num_reactions` we have `2806 + 6607 + 7123 = 16,536`
