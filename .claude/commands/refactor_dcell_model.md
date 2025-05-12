## Goal

We are refactoring pipeline for training the Dcell Model. We want to base it off the structure of the recent Dango Model Pipeline. The issue is that we have had a decent amount of updates since working with this code.

## DCell Files

These are all of the previous Dcell files. They were written 17 months ago and a lot has changed.

torchcell/datasets/dcell.py
torchcell/losses/dcell.py
torchcell/models/dcell.py
torchcell/datamodules/dcell.py
torchcell/trainers/dcell_regression.py

We know that this was working before. We ran an update the details of which can be found here. [[2025.05.09 - Reviving Model|dendron://torchcell/torchcell.models.dcell#20250509---reviving-model]]

notes/torchcell.models.dcell.md

## Understanding DCell

Here is the paper:

dcell.md

The important thing to know about DCell is that perturbations are applied by perturbing the GO structure.  One of the major updates we want to make in implementing the model is to use our updated perspective on graph perturbation. We want to load the gene ontology graph as a pyg obj and apply perturbations directly onto that obj. We don't want to have to apply deletion onto nx.Graph then convert nx.Graph to pyg Data obj. This creates too much overhead.

To make changes directly to the gene ontology incidence graph we need to write a `GraphProcessor` that can apply the `DCell` deletion perturbation to the gene Ontology Graph. I want to do this by representing the gene_ontology as a bipartite graph similar to how we do for metabolism.

torchcell/data/graph_processor.py

## All Dataset Use Neo4j Cell Now

torchcell/data/neo4j_cell.py
torchcell/datamodules/cell.py
torchcell/datamodules/perturbation_subset.py

We want train model on same dataset torchcell/scratch/load_batch_005.py as dango.

## Use Dango Pipeline To See A Recent Example of Training Different Model

torchcell/models/dango.py
torchcell/trainers/int_dango.py
experiments/005-kuzmin2018-tmi/scripts/dango.py

## Data To Test Model

torchcell/scratch/load_batch_005.py
notes/torchcell.scratch.load_batch_005.md

## Tasks

We will go step by step and not move on until I tell you.

1. Update torchcell/data/cell_data.py to add `gene_ontology` write a new `GraphProcessor` for DCell that operates on gene_ontology. Then finally more configs for load_sample_data_batch in torchcell/scratch/load_batch_005.py. We want to include graph_names list as args and incidence_graphs as args. Then we also want to include the `GraphProcessor`... In fact it would be easier if we just define a small map at the top. "dango_string9_1" with current settings, then add "dcell". Make sure you apply all of the filters. filters shown below.

Basically we need another condition gene_ontology here.

``` python
if incidence_graphs is not None:
    # Process bipartite representation only
    if "metabolism_bipartite" in incidence_graphs:
        bipartite = incidence_graphs["metabolism_bipartite"]
        _process_metabolism_bipartite(hetero_data, bipartite, node_idx_mapping)
```

```python
G = filter_by_date(G, "2017-07-19")
print(f"After date filter: {G.number_of_nodes()}")
G = filter_go_IGI(G)
print(f"After IGI filter: {G.number_of_nodes()}")
G = filter_redundant_terms(G)
print(f"After redundant filter: {G.number_of_nodes()}")
G = filter_by_contained_genes(G, n=2, gene_set=gene_set)
print(f"After containment filter: {G.number_of_nodes()}")
```

[[2025.05.10 - Inspecting Data in GoGraph|dendron://torchcell/torchcell.models.dcell#20250510---inspecting-data-in-gograph]]

2. Rewrite torchcell/models/dcell.py such that it uses torchcell/scratch/load_batch_005.py just like main torchcell/models/dango.py does. We want to be able to load 005 batch data and over fit a batch in main. Remember we want to apply perturbations directly to torch data.
3. Write torchcell/trainers/int_dango.py it should be work similarly to torchcell/trainers/int_dango.py . It should log all same metrics. Keep the same formatting, styling, etc.
4. Write experiments/005-kuzmin2018-tmi/scripts/dcell.py similar to experiments/005-kuzmin2018-tmi/scripts/dango.py .
5. write experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi.yaml ... for now i just copied experiments/005-kuzmin2018-tmi/conf/dango_kuzmin2018_tmi.yaml in it's place.

***

Wait for my command.
