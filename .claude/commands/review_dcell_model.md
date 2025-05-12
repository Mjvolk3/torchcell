## Goal

We have refactored the pipeline for training the Dcell Model. We have based it off the recent Dango Model Pipeline.

## DCell Old Files

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

## Updated DCell

This is the updated DCell files:

torchcell/models/dcell_new.py
torchcell/losses/dcell_new.py

It is imperative that we stick to to the DCell model as reported. We are not trying to make updates to the model here. We are trying to faithfully represent it in our framework.

## All Dataset Use Neo4j Cell Now

torchcell/data/neo4j_cell.py
torchcell/datamodules/cell.py
torchcell/datamodules/perturbation_subset.py
torchcell/data/graph_processor.py

We train model on same dataset torchcell/scratch/load_batch_005.py as dango.

## Use Dango Pipeline To See A Recent Example of Training Different Model

torchcell/models/dango.py
torchcell/trainers/int_dango.py
experiments/005-kuzmin2018-tmi/scripts/dango.py

## Data To Test Model

torchcell/scratch/load_batch_005.py
notes/torchcell.scratch.load_batch_005.md

## Tasks

We are trying to investigate why the model cannot overfit on one batch.

## 2025.05.12 - After you Reviewed These are Best Things to Try

- Disable BatchNorm
  - Freeze BatchNorm statistics during training
  - Replace BatchNorm with LayerNorm
  - Try removing normalization entirely to test effect
- Increase Model Capacity
  - Increase minimum output size (default 20)
  - Increase multiplier for subsystem size (default 0.3)
  - Test different neuron counts per subsystem
- Adjust Loss Weighting
  - Reduce α value (default 0.3) to focus more on primary loss
  - Try training with auxiliary losses disabled (α = 0)
  - Implement gradual α reduction during training
- Train Longer
  - Increase max epochs (500+)
  - Monitor loss curve for continued improvement
  - Implement learning rate reduction over time
