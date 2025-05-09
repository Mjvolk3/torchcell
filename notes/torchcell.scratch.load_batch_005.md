---
id: luypd6agnx9m9k59xm50sis
title: Load_batch_005
desc: ''
updated: 1746576023887
created: 1746153964493
---
We changed to coo formatting on phenotypes.

```python
dataset.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  (gene, string9_1_coexpression, gene)={
    edge_index=[2, 320295],
    num_edges=320295,
  },
  (gene, string9_1_cooccurence, gene)={
    edge_index=[2, 9266],
    num_edges=9266,
  },
  (gene, string9_1_database, gene)={
    edge_index=[2, 40093],
    num_edges=40093,
  },
  (gene, string9_1_experimental, gene)={
    edge_index=[2, 226057],
    num_edges=226057,
  },
  (gene, string9_1_fusion, gene)={
    edge_index=[2, 7965],
    num_edges=7965,
  },
  (gene, string9_1_neighborhood, gene)={
    edge_index=[2, 52208],
    num_edges=52208,
  }
)
dataset[0]
HeteroData(
  gene={
    num_nodes=6607,
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    mask=[6607],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  }
)
batch
HeteroDataBatch(
  gene={
    num_nodes=13214,
    perturbed_genes=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    pert_mask=[13214],
    mask=[13214],
    phenotype_values=[2],
    phenotype_type_indices=[2],
    phenotype_sample_indices=[2],
    phenotype_types=[2],
    phenotype_stat_values=[2],
    phenotype_stat_type_indices=[2],
    phenotype_stat_sample_indices=[2],
    phenotype_stat_types=[2],
    batch=[13214],
    ptr=[3],
  }
)
```
