---
id: luypd6agnx9m9k59xm50sis
title: Load_batch_005
desc: ''
updated: 1749709173816
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

## 2025.05.10 DCell Inclusion

```python
print("\n--- Testing DCell Configuration ---")
dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
    batch_size=2, num_workers=2, config="dcell", is_dense=False
)
dataset[0]
```

```python
dataset[0]
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    perturbation_indices_batch=[3],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  },
  gene_ontology={
    num_nodes=2242,
    node_ids=[2242],
    x=[2242, 1],
    mutant_states={
      0=dict(len=2241),
    },
    go_to_genes=dict(len=2241),
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 2796],
    num_edges=2796,
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 24890],
    num_edges=24890,
  }
)
```

The issue is that `pyg` won't be able to batch `mutant_states` since it is dict..

```python
dataset[0]
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    perturbation_indices_batch=[3],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  },
  gene_ontology={
    num_nodes=2242,
    node_ids=[2242],
    x=[2242, 1],
    mutant_states={
      0=dict(len=2241),
    },
    go_to_genes=dict(len=2241),
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 2796],
    num_edges=2796,
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 24890],
    num_edges=24890,
  }
)
datasest[0]['gene_ontology'].x.size()
Traceback (most recent call last):
  File "<string>", line 1, in <module>
NameError: name 'datasest' is not defined
dataset[0]['gene_ontology'].x.size()
torch.Size([2242, 1])
dataset[0]['gene_ontology'].x
tensor([[11.],
        [27.],
        [ 6.],
        ...,
        [ 9.],
        [ 2.],
        [ 0.]])
dataset[0]['gene_ontology'].x.max()
tensor(2342.)
```

The reason we originally did this was because we didn't want the expansion over every single named term in the printout and we didn't know how to use pyg `Data` very well.

```python
batch
HeteroDataBatch(
  gene={
    num_nodes=13214,
    node_ids=[2],
    x=[13214, 0],
    perturbed_genes=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    pert_mask=[13214],
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
  },
  gene_ontology={
    num_nodes=4484,
    node_ids=[2],
    x=[4484, 1],
    mutant_states={
      0=dict(len=2241),
    },
    go_to_genes=dict(len=2241),
    batch=[4484],
    ptr=[3],
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 5592],
    num_edges=[2],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 49780],
    num_edges=[2],
  }
)
batch['gene_ontology'].mutant_states
{0: {0: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.]), 1: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     ..., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), 775: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     ..., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), 3: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1.]), 790: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.]), 582: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     ..., 1., 1., 1., 1., 1., 1.,
        1., 1.]), 5: tensor([1., 1.]), 6: tensor([1., 1., 1., 1., 1., 1., 1., 1.]), 7: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), 1775: tensor([1., 1.]), 9: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     ..., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), 1185: tensor([1., 1., 1., 1.]), 10: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     ..., 1., 1., 1., 1., 1., 1.,
        1., 1.]), 12: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), 14: tensor([1., 1.]), 696: tensor([1., 1., 1., 1.]), 1191: tensor([1., 1., 1., 1.]), 15: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1.]), 17: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), ...}}
len(batch['gene_ontology'].mutant_states)
1
len(batch['gene_ontology'].mutant_states[0])
2241
len(dataset[0]['gene_ontology'].mutant_states)
1
len(dataset[0]['gene_ontology'].mutant_states[0])
2241
```

I want to avoid just putting all states in 'gene_ontology' directly because it will print everything. Instead I want a more nested structures. I think that we make gene ontology a pyg `Data` object then have named mutant states for all gene_ontology terms.

## 2025.05.10 - DCell No Dicts

```python
dataset.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  gene_ontology={
    num_nodes=2242,
    node_ids=[2242],
    x=[2242, 1],
    term_gene_mapping=[24890, 2],
    term_gene_counts=[2242],
    term_to_gene_dict=dict(len=2242),
    max_genes_per_term=2342,
    term_ids=[2242],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 24890],
    num_edges=24890,
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 2796],
    num_edges=2796,
  }
)
dataset[0]
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    perturbation_indices_batch=[3],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  },
  gene_ontology={
    num_nodes=2242,
    node_ids=[2242],
    x=[2242, 1],
    mutant_state=[24890, 3],
    term_ids=[2242],
    max_genes_per_term=2342,
    term_gene_mapping=[24890, 2],
    term_gene_counts=[2242],
    term_to_gene_dict=dict(len=2242),
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 2796],
    num_edges=2796,
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 24890],
    num_edges=24890,
  }
)
batch
HeteroDataBatch(
  gene={
    num_nodes=13214,
    node_ids=[2],
    x=[13214, 0],
    perturbed_genes=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    pert_mask=[13214],
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
  },
  gene_ontology={
    num_nodes=4484,
    node_ids=[2],
    x=[4484, 1],
    mutant_state=[49780, 3],
    mutant_state_batch=[49780],
    mutant_state_ptr=[3],
    term_ids=[2],
    max_genes_per_term=[2],
    term_gene_mapping=[49780, 2],
    term_gene_counts=[4484],
    term_to_gene_dict=dict(len=2242),
    batch=[4484],
    ptr=[3],
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 5592],
    num_edges=[2],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 49780],
    num_edges=[2],
  }
)
```

The first index in `GO` the second is gene and the last is the mutant state.

```python
dataset[0]['gene_ontology'].mutant_state[:,0].max()
tensor(2240.)
dataset[0]['gene_ontology'].mutant_state[:,1].max()
tensor(6606.)
dataset[0]['gene_ontology'].mutant_state[:,2].max()
tensor(1.)
```

Testing from the repl. All perturbed genes have bits flipped to 0.

```python
# Get the perturbed gene indices
perturbed_genes = dataset[0]['gene']['perturbation_indices']
print(f"Perturbed gene indices: {perturbed_genes}")

# Check each perturbed gene
for gene_idx in perturbed_genes:
    # Find matches in the mutant_state tensor (second column has gene indices)
    matches = (dataset[0]['gene_ontology'].mutant_state[:, 1] == gene_idx).nonzero()
    
    # Handle matches appropriately
    if matches.numel() == 0:
        print(f"\nGene {gene_idx} appears in 0 GO terms")
        continue
    
    # Get indices as a list to avoid dimension issues
    match_indices = matches.view(-1).tolist()
    
    # Get GO term indices from the first column of mutant_state
    go_term_indices = dataset[0]['gene_ontology'].mutant_state[match_indices, 0].tolist()
    
    # Get the actual GO terms
    go_terms = [dataset[0]['gene_ontology'].term_ids[int(idx)] for idx in go_term_indices]
    
    print(f"\nGene {gene_idx} appears in {len(match_indices)} GO terms:")
    print(f"GO term indices: {go_term_indices}")
    print(f"GO terms: {go_terms}")
    
    # Check if the mutant state is set to 0 (bit flipped)
    flipped_state = dataset[0]['gene_ontology'].mutant_state[match_indices, 2]
    print(f"Mutant states: {flipped_state}")
    print(f"All bits correctly flipped to 0: {bool((flipped_state == 0).all())}")

# Create a mask for all mutant states that do NOT correspond to perturbed genes
mask = torch.ones(dataset[0]['gene_ontology'].mutant_state.shape[0], dtype=torch.bool)

# Exclude entries for perturbed genes
for gene_idx in perturbed_genes:
    gene_mask = dataset[0]['gene_ontology'].mutant_state[:, 1] == gene_idx
    mask &= ~gene_mask

# Check the values in position 2 for non-perturbed entries
non_perturbed_states = dataset[0]['gene_ontology'].mutant_state[mask, 2]

# Verify all are 1
all_ones = (non_perturbed_states == 1).all()
print(f"\nTotal mutant_state entries: {dataset[0]['gene_ontology'].mutant_state.shape[0]}")
print(f"Entries corresponding to perturbed genes: {dataset[0]['gene_ontology'].mutant_state.shape[0] - mask.sum()}")
print(f"Entries corresponding to non-perturbed genes: {mask.sum()}")
print(f"All non-perturbed entries have state 1: {bool(all_ones)}")

# If not all ones, count how many are not 1
if not all_ones:
    non_ones_count = (non_perturbed_states != 1).sum()
    print(f"Number of non-perturbed entries with state ≠ 1: {non_ones_count}")
    
    # Sample a few entries that are not 1
    non_ones_indices = (non_perturbed_states != 1).nonzero().view(-1)[:5]
    print(f"Sample indices with state ≠ 1: {non_ones_indices}")
    print(f"Sample values: {non_perturbed_states[non_ones_indices]}")
Perturbed gene indices: tensor([ 861, 4546, 6502])

Gene 861 appears in 8 GO terms:
GO term indices: [615.0, 1219.0, 1341.0, 1345.0, 1346.0, 1588.0, 285.0, 184.0]
GO terms: ['GO:0006366', 'GO:0030174', 'GO:0031509', 'GO:0031571', 'GO:0031573', 'GO:0042138', 'GO:0003688', 'GO:0000785']
Mutant states: tensor([0., 0., 0., 0., 0., 0., 0., 0.])
All bits correctly flipped to 0: True

Gene 4546 appears in 1 GO terms:
GO term indices: [478.0]
GO terms: ['GO:0005783']
Mutant states: tensor([0.])
All bits correctly flipped to 0: True

Gene 6502 appears in 2 GO terms:
GO term indices: [278.0, 478.0]
GO terms: ['GO:0003674', 'GO:0005783']
Mutant states: tensor([0., 0.])
All bits correctly flipped to 0: True

Total mutant_state entries: 24890
Entries corresponding to perturbed genes: 11
Entries corresponding to non-perturbed genes: 24879
All non-perturbed entries have state 1: True
```

## 2025.05.20 - DCell Mutant State with Strata

Indices of `mutant_state`:

1. GO
2. gene
3. strata
4. state `[0,1]`

```python
dataset_unfiltered[0]
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    perturbation_indices_batch=[3],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    mutant_state=[59986, 5],
    term_ids=[2655],
    max_genes_per_term=2444,
    term_gene_mapping=[59986, 2],
    term_gene_counts=[2655],
    term_to_gene_dict=dict(len=2655),
    strata=[2655],
    stratum_to_terms=dict(len=13),
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 3208],
    num_edges=3208,
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 59986],
    num_edges=59986,
  }
)
batch_unfiltered
HeteroDataBatch(
  gene={
    num_nodes=13214,
    node_ids=[2],
    x=[13214, 0],
    perturbed_genes=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    pert_mask=[13214],
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
  },
  gene_ontology={
    num_nodes=5310,
    node_ids=[2],
    x=[5310, 1],
    mutant_state=[119972, 5],
    mutant_state_batch=[119972],
    mutant_state_ptr=[3],
    term_ids=[2],
    max_genes_per_term=[2],
    term_gene_mapping=[119972, 2],
    term_gene_counts=[5310],
    term_to_gene_dict=dict(len=2655),
    strata=[5310],
    stratum_to_terms=dict(len=13),
    batch=[5310],
    ptr=[3],
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 6416],
    num_edges=[2],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 119972],
    num_edges=[2],
  }
)
```

## 2025.05.20 - DCell with Message Passing Data

```python
dataset_unfiltered.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    term_gene_mapping=[59986, 2],
    term_gene_counts=[2655],
    term_to_gene_dict=dict(len=2655),
    term_ids=[2655],
    strata=[2655],
    stratum_to_terms=dict(len=13),
    term_gene_edges=[2, 59986],
    term_gene_attrs=[59986, 2],
    output_dimensions=dict(len=2655),
    topo_order=[2655],
    children_dict=dict(len=1128),
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 59986],
    num_edges=59986,
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 3208],
    num_edges=3208,
  }
)
dataset_unfiltered[0]
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    perturbation_batch=[3],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    strata=[2655],
    topo_order=[2655],
    term_gene_counts=[2655],
    term_gene_mapping=[59986, 2],
    term_gene_edges=[2, 59986],
    term_gene_attrs=[59986, 2],
    strata_idx=[2655],
    terms_by_strata=[2655],
    child_parent_idx=[2, 3208],
    output_dims=[2655],
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 3208],
    num_edges=3208,
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 59986],
    num_edges=59986,
  }
)
batch_unfiltered
HeteroDataBatch(
  gene={
    num_nodes=13214,
    node_ids=[2],
    x=[13214, 0],
    perturbed_genes=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    pert_mask=[13214],
    perturbation_batch=[6],
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
  },
  gene_ontology={
    num_nodes=5310,
    node_ids=[2],
    x=[5310, 1],
    strata=[5310],
    topo_order=[5310],
    term_gene_counts=[5310],
    term_gene_mapping=[119972, 2],
    term_gene_edges=[4, 59986],
    term_gene_attrs=[119972, 2],
    strata_idx=[5310],
    terms_by_strata=[5310],
    child_parent_idx=[4, 3208],
    output_dims=[5310],
    batch=[5310],
    ptr=[3],
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 6416],
    num_edges=[2],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 119972],
    num_edges=[2],
  }
)
```

how perturbations are applied.

```python
dataset_unfiltered.cell_graph['gene_ontology']['term_gene_attrs'].sum(0)
tensor([225014.,  59986.])
dataset_unfiltered[0]['gene_ontology']['term_gene_attrs'].sum(0)
tensor([225014.,  59950.])
```

## 2025.05.21 - Data Structure for Pyg Message Passing

```python
dataset_unfiltered.cell_graph
dataset_unfiltered[0]
batch_unfiltered
batch_unfiltered['gene_ontology'].go_gene_state_ptr
```

```python
dataset_unfiltered.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    go_gene_state=[59986, 3],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 59986],
    num_edges=59986,
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 3208],
    num_edges=3208,
  }
)
dataset_unfiltered[0]
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    perturbation_batch=[3],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    go_gene_state=[59986, 3],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 59986],
    num_edges=59986,
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 3208],
    num_edges=3208,
  }
)
batch_unfiltered
HeteroDataBatch(
  gene={
    num_nodes=13214,
    node_ids=[2],
    x=[13214, 0],
    perturbed_genes=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    pert_mask=[13214],
    perturbation_batch=[6],
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
  },
  gene_ontology={
    num_nodes=5310,
    node_ids=[2],
    x=[5310, 1],
    go_gene_state=[119972, 3],
    go_gene_state_batch=[119972],
    go_gene_state_ptr=[3],
    batch=[5310],
    ptr=[3],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 119972],
    num_edges=[2],
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 6416],
    num_edges=[2],
  }
)
batch_unfiltered['gene_ontology'].go_gene_state_ptr
tensor([     0,  59986, 119972])
```

## 2025.05.21 - Data Structure Don't Propagate Edge Info

`go_gene_state` is a matrix with first column being `go:int`, second column is `gene:int` and last column is `state:int`. State is 1 if not perturbed.

```python
dataset_unfiltered.cell_graph
dataset_unfiltered[0]
batch_unfiltered
batch_unfiltered['gene_ontology'].go_gene_state_ptr
```

```python
dataset_unfiltered.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    go_gene_state=[59986, 3],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 59986],
    num_edges=59986,
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 3208],
    num_edges=3208,
  }
)
dataset_unfiltered[0]
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    perturbation_batch=[3],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    go_gene_state=[59986, 3],
  }
)
batch_unfiltered
HeteroDataBatch(
  gene={
    num_nodes=13214,
    node_ids=[2],
    x=[13214, 0],
    perturbed_genes=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    pert_mask=[13214],
    perturbation_batch=[6],
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
  },
  gene_ontology={
    num_nodes=5310,
    node_ids=[2],
    x=[5310, 1],
    go_gene_state=[119972, 3],
    go_gene_state_batch=[119972],
    go_gene_state_ptr=[3],
    batch=[5310],
    ptr=[3],
  }
)
batch_unfiltered['gene_ontology'].go_gene_state_ptr
tensor([     0,  59986, 119972])
```

## 2025.05.21 - DCell Data Reversion

```python
dataset_unfiltered.cell_graph
dataset_unfiltered[0]
batch_unfiltered
batch_unfiltered['gene_ontology'].mutant_state_ptr
```

```python
dataset_unfiltered.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    term_gene_mapping=[59986, 2],
    term_gene_counts=[2655],
    term_to_gene_dict=dict(len=2655),
    max_genes_per_term=2444,
    term_ids=[2655],
    strata=[2655],
    stratum_to_terms=dict(len=13),
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 59986],
    num_edges=59986,
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 3208],
    num_edges=3208,
  }
)
dataset_unfiltered[0]
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    perturbation_indices_batch=[3],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    mutant_state=[59986, 5],
    term_ids=[2655],
    max_genes_per_term=2444,
    term_gene_mapping=[59986, 2],
    term_gene_counts=[2655],
    term_to_gene_dict=dict(len=2655),
    strata=[2655],
    stratum_to_terms=dict(len=13),
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 3208],
    num_edges=3208,
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 59986],
    num_edges=59986,
  }
)
batch_unfiltered
HeteroDataBatch(
  gene={
    num_nodes=13214,
    node_ids=[2],
    x=[13214, 0],
    perturbed_genes=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    pert_mask=[13214],
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
  },
  gene_ontology={
    num_nodes=5310,
    node_ids=[2],
    x=[5310, 1],
    mutant_state=[119972, 5],
    mutant_state_batch=[119972],
    mutant_state_ptr=[3],
    term_ids=[2],
    max_genes_per_term=[2],
    term_gene_mapping=[119972, 2],
    term_gene_counts=[5310],
    term_to_gene_dict=dict(len=2655),
    strata=[5310],
    stratum_to_terms=dict(len=13),
    batch=[5310],
    ptr=[3],
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 6416],
    num_edges=[2],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 119972],
    num_edges=[2],
  }
)
batch_unfiltered['gene_ontology'].mutant_state_ptr
tensor([     0,  59986, 119972])
```

## 2025.05.21 - The Return of go_gene_strata_state

```python
dataset_unfiltered.cell_graph
dataset_unfiltered[0]
batch_unfiltered
batch_unfiltered['gene_ontology'].go_gene_strata_state_ptr
```

```python
dataset_unfiltered.cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    x=[2655, 1],
    term_gene_mapping=[59986, 2],
    term_gene_counts=[2655],
    term_to_gene_dict=dict(len=2655),
    max_genes_per_term=2444,
    term_ids=[2655],
    strata=[2655],
    stratum_to_terms=dict(len=13),
    go_gene_strata_state=[59986, 4],
  },
  (gene, has_annotation, gene_ontology)={
    edge_index=[2, 59986],
    num_edges=59986,
  },
  (gene_ontology, is_child_of, gene_ontology)={
    edge_index=[2, 3208],
    num_edges=3208,
  }
)
dataset_unfiltered[0]
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    perturbation_indices_batch=[3],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  },
  gene_ontology={
    num_nodes=2655,
    node_ids=[2655],
    go_gene_strata_state=[59986, 4],
  }
)
batch_unfiltered
HeteroDataBatch(
  gene={
    num_nodes=13214,
    node_ids=[2],
    x=[13214, 0],
    perturbed_genes=[2],
    perturbation_indices=[6],
    perturbation_indices_batch=[6],
    perturbation_indices_ptr=[3],
    pert_mask=[13214],
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
  },
  gene_ontology={
    num_nodes=5310,
    node_ids=[2],
    go_gene_strata_state=[119972, 4],
    go_gene_strata_state_batch=[119972],
    go_gene_strata_state_ptr=[3],
    batch=[5310],
    ptr=[3],
  }
)
batch_unfiltered['gene_ontology'].go_gene_strata_state_ptr
tensor([     0,  59986, 119972])
```

## 2025.05.22 - Size Mismatch Across Batch Issue

We show there should be no issue.

```python
dataset_unfiltered.cell_graph['gene_ontology'].go_gene_strata_state[:,:3].flatten().size()
torch.Size([179958])
(dataset_unfiltered.cell_graph['gene_ontology'].go_gene_strata_state[:,:3].flatten()== dataset_unfiltered[0]['gene_ontology'].go_gene_strata_state[:,:3].flatten()).sum()
tensor(179958)
[(dataset_unfiltered.cell_graph['gene_ontology'].go_gene_strata_state[:,:3].flatten()== dataset_unfiltered[i]['gene_ontology'].go_gene_strata_state[:,:3].flatten()).sum() for i in range(32)]
[tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958), tensor(179958)]
```

## 2025.06.12 - Showing Values for Data When COO Format

```python
dataset_hetero.cell_graph
dataset_hetero[0]
batch_hetero
```

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
