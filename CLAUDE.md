## For Local Configs

`/Users/michaelvolk/Documents/projects/torchcell/CLAUDE.local.md`

- Includes how to run local python

## Programming Guide

- Do NOT ever use fallback mechanisms unless we clearly tell you to. This means minimize try except blocks, unnecessary conditionals, etc.

## Dendron Paths

We use dendron paths in markdown files to link between project notes. These are very useful for giving additional context.

File of these patters:

[[2025.05.10 - Inspecting Data in GoGraph|dendron://torchcell/torchcell.models.dcell#20250510---inspecting-data-in-gograph]]

Can be found here:

notes/torchcell.models.dcell.md

**The General Pattern**

`notes/` dir is skipped in path description as it is default location

Dendron encode from `torchcell/torchcell.models.dcell` to `notes/torchcell.models.dcell.md`

When I tell you to write some output to a file that is in `notes/` then typially you just need to append or modify, we don't want you messing up dendron frontmatter.

## Saving Images in Python

All images should be saved in `ASSET_IMAGES_DIR`

Do this by using `load_dotenv` and time stamp the images with by using torchcell/timestamp.py

The common patters is `(osp.join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png"))`

## When Creating or Designing new Experiments

These env vars will be read in in python with:

```python
from dotenv import load_dotenv
```

These are the most relevant dirs. (These are just example paths, they can change so just read in the variable.):

```bash
DATA_ROOT="/scratch/projects/torchcell-scratch"
ASSET_IMAGES_DIR="/home/michaelvolk/Documents/projects/torchcell/notes/assets/images"
EXPERIMENT_ROOT="/home/michaelvolk/Documents/projects/torchcell/experiments"
```

**`DATA_ROOT`**

- Where all of the large data is saved becuase this dir is typically mapped to larger storage.
- Datasets for training
- Datasets used for inference
- Inference results too are often put here because the outputs can be very large.

**`ASSET_IMAGES_DIR`**

- Where we want to save all image output
- Often convenient if there are more than one image for that given experiment to put them into dir with something like `osp.join(ASSET_IMAGES_DIR, f"006-kuzmin-tmi-inferece_1_{timestamp}")`. Often we like to start by default putting time stamps on images and we iterate to a decent idea, then if we want things to stably overwrite we will ask to remove `timestamp` after. So add by default.

**`EXPERIMENT_ROOT`**

This is what a typical experiment looks like. There is configuration in `conf/` that parameterizes experiements written in `scripts`. Slurm scripts are also in scripts and they typically have matching names with their experiments. Inside naming is also typically a designation of which slurm machine or partition a given scripts belongs to. We try to copy slurm scripts are typically numbered in a fashion that matches yamls to have a very transparaent experiment ecosystem. If we are running sweeps with optuna or wandb. we break from this pattern because we only need one config specifying the script - so we can many outputs for one script instead of one-to-one. Sometimes we inspect models here and dump `profile_results/`. `slurm/outputs` is resereved for concatenated slurm output and error. `queries/` holds the queries that the experiment used for the study.

```bash
/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi
├── conf
├── profiling_results
├── queries
├── results
├── scripts
└── slurm

6 directories, 3 files
```

Here is more depth on a more recent experiment. We cannot show for previous becuase it is too deep.

```bash
(base) michaelvolk@gilahyper torchcell % tree /home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi                       16:57
/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi
├── conf
│   ├── default.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000_inference_r0.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000_inference_r1.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000_inference_r2.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_002.yaml
│   ├── equivariant_cell_graph_transformer_mmli_001.yaml
│   └── equivariant_cell_graph_transformer_mmli_003.yaml
├── queries
│   └── 001_small_build.cql
├── results
├── scripts
│   ├── equivariant_cell_graph_transformer_inference_1.py
│   ├── equivariant_cell_graph_transformer.py
│   ├── gh_equivariant_cell_graph_transformer_inference_1_E006_M01.slurm
│   ├── gh_equivariant_cell_graph_transformer_inference_1_E006_M02.slurm
│   ├── gh_equivariant_cell_graph_transformer_inference_1_E006_M03.slurm
│   ├── gh_select_12_and_24_gene_top_triples.slurm
│   ├── igb_cabbi_equivariant_cell_graph_transformer-ddp_000.slurm
│   ├── igb_cabbi_equivariant_cell_graph_transformer-ddp_002.slurm
│   ├── igb_mmli_equivariant_cell_graph_transformer-ddp_001.slurm
│   ├── igb_mmli_equivariant_cell_graph_transformer-ddp_003.slurm
│   ├── inference_dataset_1.py
│   ├── __init__.py
│   ├── __pycache__
│   │   └── inference_dataset_1.cpython-313.pyc
│   ├── query.py
│   └── select_12_and_24_genes_top_triples.py
└── slurm
    └── output
        ├── 010-inference-M01_686.out
        ├── 010-inference-M02_690.out
        ├── 010-inference-M02_696.out
        ├── 010-inference-M02_698.out
        ├── 010-inference-M03_693.out
        ├── 010-inference-M03_697.out
        ├── 010-inference-M03_699.out
        └── 010-select-12-genes_694.out
```
