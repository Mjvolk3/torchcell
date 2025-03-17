---
id: tgje10ghnphks3tfzjdbjlw
title: 203022-HeteroNSA-requirement
desc: ''
updated: 1742020996046
created: 1741915835262
---
The HeteroNSA is composed of interleaved Self Attention Blocks (SAB) and Masked Attention Blocks (MAB). We are using this model to replace a `HeteroConv` over graphs. MAB uses a boolean mask to keep edges for attention, and SAB computes attention across all nodes.

This is what data will look like.

```python
dataset.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
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
    edge_index=[2, 144211],
    num_edges=144211,
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16101],
    num_edges=16101,
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
    x=[6607, 0],
    x_pert=[2, 0],
    fitness=[1],
    fitness_std=[1],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
    pert_mask=[6607],
    fitness_original=[1],
    gene_interaction_original=[1],
    mask=[6607],
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
    edge_index=[2, 144102],
    num_edges=144102,
    pert_mask=[144211],
    adj_mask=[6607, 6607],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16096],
    num_edges=16096,
    pert_mask=[16101],
    adj_mask=[6607, 6607],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5450],
    num_edges=5450,
    pert_mask=[5450],
    inc_mask=[6607, 7122],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 13580],
    stoichiometry=[13580],
    edge_type=[13580],
    num_edges=13580,
    pert_mask=[13580],
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
    inc_mask=[7122, 2806],
  }
)
```

```python
batch
HeteroDataBatch(
  gene={
    node_ids=[2],
    num_nodes=13209,
    ids_pert=[2],
    cell_graph_idx_pert=[5],
    x=[13214, 0],
    x_batch=[13214],
    x_ptr=[3],
    x_pert=[5, 0],
    x_pert_batch=[5],
    x_pert_ptr=[3],
    fitness=[2],
    fitness_std=[2],
    gene_interaction=[2],
    gene_interaction_p_value=[2],
    pert_mask=[13214],
    fitness_original=[2],
    gene_interaction_original=[2],
    mask=[13214],
    batch=[13209],
    ptr=[3],
  },
  reaction={
    num_nodes=14238,
    node_ids=[2],
    pert_mask=[14244],
    mask=[14244],
    batch=[14238],
    ptr=[3],
  },
  metabolite={
    node_ids=[2],
    num_nodes=5612,
    pert_mask=[5612],
    mask=[5612],
    batch=[5612],
    ptr=[3],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 288267],
    num_edges=[2],
    pert_mask=[288422],
    adj_mask=[13214, 6607],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 32196],
    num_edges=[2],
    pert_mask=[32202],
    adj_mask=[13214, 6607],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 10876],
    num_edges=[2],
    pert_mask=[10900],
    inc_mask=[13214, 7122],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 27148],
    stoichiometry=[27148],
    edge_type=[27148],
    num_edges=[2],
    pert_mask=[27160],
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
    inc_mask=[14244, 2806],
  }
)
```

Saving `dataset[0]` costs `190M`. We can just call it `200M`.

HeteroNSA Requirements

- [ ] We need absolutely need to use flex attention due to memory savings. It has issues on cpu so we can implement fallback for `cpu` for local `cpu` testing, but on gpu if flex attention isn't working we should NOT fallback `cpu` implementation and it should fail.
- [ ] Handled Directed and Undirected graphs. This should just automatically be handled by non-symmetric attention matrices.
- [ ] Boolean masks for dense adj. If we don't do this we can only batch 2 instances before memory issues. This is a huge issue so we must use Boolean masks.
- [ ] In `NSA` block we must include `edge_attrs` but casting them to dense is expensive since these are float. Instead we must keep them in sparse (normal) pyg format.
- [ ] We need to make sure `Stoichiometry` is being used as `edge_attrs` in `rmr`. We should include edge attrs as linear projection of edge attrs that are added to attention weights.

Modules work in progress:

- `torchcell/nn/self_attention_block.py`
- `torchcell/nn/masked_attention_block.py`
- `torchcell/nn/nsa_encoder.py`
- `torchcell/nn/hetero_nsa.py`
- `torchcell/models/hetero_cell_nsa.py`

Modules tests work in progress:

- `tests/torchcell/nn/test_self_attention_block.py`
- `tests/torchcell/nn/test_masked_attention_block.py`
- `tests/torchcell/nn/test_nsa_encoder.py`
- `tests/torchcell/nn/test_hetero_nsa.py`
- `tests/torchcell/models/test_hetero_cell_nsa.py`
