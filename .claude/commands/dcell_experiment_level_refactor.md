## Background

We have written the experimental pipeline for `Dcell`. It is very similar to the pipeline for the `Dango` model. have kept it similar to the Dango pipeline as possible. We kept a record or all of the same metrics, use the same code structure, etc.

To be able to do this we had to create a go process in to_cell and we had to modify the graph processor to propagate gene deletions on the `Dcell` graph.

## Main Issue

Right now the pipeline runs but it is extremely slow and doesn't utilize the GPU well. For instance it has taken 2 full days to get through 100,000 data samples for just a single epoch. This is due to the DAG nature of the pipeline processing submodules one at a time.

/Users/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py (Line:828)

```python
def forward(
  self, subsystem_outputs: Dict[str, torch.Tensor]
  ) -> Dict[str, torch.Tensor]:
  """
  Forward pass applying linear transformation to each subsystem output.

  Args:
      subsystem_outputs: Dictionary mapping subsystem names to their outputs

  Returns:
      Dictionary mapping subsystem names to transformed outputs
  """
  linear_outputs = {}

  for subsystem_name, subsystem_output in subsystem_outputs.items():
      if subsystem_name in self.subsystem_linears:
          transformed_output = self.subsystem_linears[subsystem_name](
              subsystem_output
          )
          linear_outputs[subsystem_name] = transformed_output

  return linear_outputs
```

We should be able do better than this with recognizing the fact that each of the subsystems has an associated level. /Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/graph.py ... I am pretty sure that since the GO forms a DAG that we should be able to group operations. I am also pretty sure GO nodes on the same level don't interact but this needs to be double checked. If this is the case this means that we group all submodules by the same level, concatenate what would be all their linear layers together, bulk process them, the redistribute their outputs to their corresponding modules which are then fed as input too parent submodules. We would again group parent submodules on the same level. Rinse and repeat. This batch processing should be able to dramatically increase the efficiency of the model.

If we read [[torchcell.scratch.load_batch_005]] we can see our data has the following structure.

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

This shows that we have not included the level in the data representation. We will need to propagate this from graph.

From the `repl`

/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/graph.py (line 1403)

```python
def main() -> None:
    import os
    import random

    import matplotlib.pyplot as plt
    import pandas as pd
    from dotenv import load_dotenv

    from torchcell.datasets.scerevisiae import (
        DmfCostanzo2016Dataset,
        SmfCostanzo2016Dataset,
    )

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=True,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    ) # breakpoint here.
########## Output #########
graph.G_go.nodes['GO:0098745']
{'id': 'GO:0098745', 'item_id': 'GO:0098745', 'name': 'RNA decapping complex', 'namespace': 'cellular_component', 'level': 2, 'depth': 2, 'is_obsolete': False, 'alt_ids': {}, 'gene_set': <torchcell.sequence.data.GeneSet object at 0x328b06650>, 'genes': {'YGL222C': {...}, 'YNL118C': {...}, 'YOL149W': {...}}}
special variables
function variables
'id' =
'GO:0098745'
'item_id' =
'GO:0098745'
'name' =
'RNA decapping complex'
'namespace' =
'cellular_component'
'level' =
2
'depth' =
2
'is_obsolete' =
False
'alt_ids' =
{}
'gene_set' =
<torchcell.sequence.data.GeneSet object at 0x328b06650>
'genes' =
{'YGL222C': {'go_details': {...}}, 'YNL118C': {'go_details': {...}}, 'YOL149W': {'go_details': {...}}}
special variables
function variables
'YGL222C' =
{'go_details': {'id': 8135561, 'annotation_type': 'manually curated', 'date_created': '2018-03-07', 'qualifier': 'part of', 'locus': {...}, 'go': {...}, 'reference': {...}, 'source': {...}, 'experiment': {...}, 'properties': [...]}}
'YNL118C' =
{'go_details': {'id': 8135567, 'annotation_type': 'manually curated', 'date_created': '2018-03-07', 'qualifier': 'part of', 'locus': {...}, 'go': {...}, 'reference': {...}, 'source': {...}, 'experiment': {...}, 'properties': [...]}}
'YOL149W' =
{'go_details': {'id': 8135594, 'annotation_type': 'manually curated', 'date_created': '2018-03-07', 'qualifier': 'part of', 'locus': {...}, 'go': {...}, 'reference': {...}, 'source': {...}, 'experiment': {...}, 'properties': [...]}}
len() =
3
len() =
10
graph.G_go.nodes['GO:ROOT']
{'name': 'GO Super Node', 'namespace': 'super_root', 'level': -1}
special variables
function variables
'name' =
'GO Super Node'
'namespace' =
'super_root'
'level' =
-1
len() =
3
```

We see that we have the relevant levels for grouping within the `graph.G_go` data structure.

## Relevant Dango Files

These are the relevant Dango files

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/dango.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/trainers/int_dango.py
/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/conf/dango_kuzmin2018_tmi.yaml

## Relevant DCell Files

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/trainers/int_dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/losses/dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/models/dcell.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/scratch/load_batch_005.py
/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi.yaml

dcell.md (the original paper.)

## Relevant Data Structure Files

/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/cell_data.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/graph_processor.py
/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/graph.py

## Tasks

- [x] Review the entire plan and criticize it. Will the idea help speed computation or not. → experiments/005-kuzmin2018-tmi/scripts/test_go_hierarchy_levels.py [[Test_go_hierarchy_levels|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.test_go_hierarchy_levels]]
- [x] Run a test on graph making sure that there are no connections within levels. We are using this test to confirm that we can group the computation per level. Put the file here /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/test_<appropriate_name>.py .
- [x] Modify torchcell/data/cell_data.py to include the level of the submodule within the `mutant_state` tensor.
- [x]  Modify `/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/graph_processor.py` to apply the perturbation to the GO graph. Nothing changes here really we just want to make sure that the level is propagated along.
- [x] Modify torchcell/models/dcell.py such that we process submodules by level.

## Extra Info

- If you want to write any additional tests put them here /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/

![](./notes/assets/images/go_batch_distribution_2025-05-18-18-11-30.png)
![](.notes/assets/images/go_level_vs_batch_comparison_2025-05-18-18-11-30.png)

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/test_go_topological_batching.py
[[Test_go_topological_batching|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.test_go_topological_batching]]

/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/test_go_hierarchy_levels.py
[[Test_go_hierarchy_levels|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.test_go_hierarchy_levels]]

***

Now we are testing Dcell
