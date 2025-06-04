---
id: sz8r7xtbyhl1r8oy5p6lpw6
title: Hetero_cell_bipartite_dango_gi
desc: ''
updated: 1748996654673
created: 1748977996389
---
## 2025.06.03 - Data Updated

`torchcell/scratch/load_batch_005.py`

```python
dataset_hetero.cell_graph
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
    w_growth=[7122],
  },
  (gene, physical, gene)={
    edge_index=[2, 144211],
    num_edges=144211,
  },
  (gene, regulatory, gene)={
    edge_index=[2, 44310],
    num_edges=44310,
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 26325],
    stoichiometry=[26325],
    num_edges=26325,
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5450],
    num_edges=4881,
  }
)
dataset_hetero[0]
HeteroData(
  gene={
    node_ids=[6604],
    num_nodes=6604,
    ids_pert=[3],
    perturbation_indices=[3],
    x=[6604, 0],
    x_pert=[3, 0],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
    pert_mask=[6607],
  },
  reaction={
    num_nodes=7122,
    node_ids=[7122],
    w_growth=[7122],
    pert_mask=[7122],
  },
  metabolite={
    node_ids=[2806],
    num_nodes=2806,
    pert_mask=[2806],
  },
  (gene, physical, gene)={
    edge_index=[2, 144100],
    num_edges=144100,
    pert_mask=[144211],
  },
  (gene, regulatory, gene)={
    edge_index=[2, 44289],
    num_edges=44289,
    pert_mask=[44310],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5450],
    num_edges=5450,
    pert_mask=[5450],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 26325],
    stoichiometry=[26325],
    num_edges=26325,
  }
)
batch_hetero
HeteroDataBatch(
  gene={
    node_ids=[2],
    num_nodes=13208,
    ids_pert=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    x=[13208, 0],
    x_pert=[6, 0],
    phenotype_values=[2],
    phenotype_type_indices=[2],
    phenotype_sample_indices=[2],
    phenotype_types=[2],
    phenotype_stat_values=[2],
    phenotype_stat_type_indices=[2],
    phenotype_stat_sample_indices=[2],
    phenotype_stat_types=[2],
    pert_mask=[13214],
    batch=[13208],
    ptr=[3],
  },
  reaction={
    num_nodes=14243,
    node_ids=[2],
    w_growth=[14243],
    pert_mask=[14244],
    batch=[14243],
    ptr=[3],
  },
  metabolite={
    node_ids=[2],
    num_nodes=5612,
    pert_mask=[5612],
    batch=[5612],
    ptr=[3],
  },
  (gene, physical, gene)={
    edge_index=[2, 288178],
    num_edges=[2],
    pert_mask=[288422],
  },
  (gene, regulatory, gene)={
    edge_index=[2, 88574],
    num_edges=[2],
    pert_mask=[88620],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 10899],
    num_edges=[2],
    pert_mask=[10900],
  },
  (reaction, rmr, metabolite)={
    hyperedge_index=[2, 52645],
    stoichiometry=[52645],
    num_edges=[2],
  }
)
```

## 2025.06.03 - Detailed View of Data For Indexing Based on Phenotype Type

For printing

```python
dataset_hetero[0]['gene'].phenotype_values
dataset_hetero[0]['gene'].phenotype_type_indices
dataset_hetero[0]['gene'].phenotype_sample_indices
dataset_hetero[0]['gene'].phenotype_types
dataset_hetero[0]['gene'].phenotype_stat_values
dataset_hetero[0]['gene'].phenotype_stat_type_indices
dataset_hetero[0]['gene'].phenotype_stat_sample_indices
dataset_hetero[0]['gene'].phenotype_stat_types
```

```python
dataset_hetero[0]['gene'].phenotype_values
tensor([-0.0588])
dataset_hetero[0]['gene'].phenotype_type_indices
tensor([0])
dataset_hetero[0]['gene'].phenotype_sample_indices
tensor([0])
dataset_hetero[0]['gene'].phenotype_types
['gene_interaction']
dataset_hetero[0]['gene'].phenotype_stat_values
tensor([0.1059])
dataset_hetero[0]['gene'].phenotype_stat_type_indices
tensor([0])
dataset_hetero[0]['gene'].phenotype_stat_sample_indices
tensor([0])
dataset_hetero[0]['gene'].phenotype_stat_types
['gene_interaction_p_value']
```
