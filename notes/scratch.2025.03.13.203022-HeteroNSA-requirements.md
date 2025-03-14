---
id: tgje10ghnphks3tfzjdbjlw
title: 203022-HeteroNSA-requirement
desc: ''
updated: 1741918640453
created: 1741915835262
---
The HeteroNSA is composed of interleaved Self Attention Blocks (SAB) and Masked Attention Blocks (MAB). We are using this model to replace a `HeteroConv` over graphs. MAB uses a boolean mask to keep edges for attention, and SAB computes attention across all nodes.

This is what data will look like.

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


HeteroNSA Requirements

- [ ] Handled Directed and Undirected graphs. This should just automatically be handled by non-symmetric attention matrices.
- [ ] Boolean masks for dense adj. If we don't do this we can only batch 2 instances before memory issues. This is a huge issue so we must use Boolean masks.
- [ ] In `NSA` block we must include `edge_attrs` but casting them to dense is expensive since these are float. Instead we must keep them in sparse (normal) pyg format.
- [ ] We need to make sure `Stoichiometry` is being used as `edge_attrs` in `rmr` 