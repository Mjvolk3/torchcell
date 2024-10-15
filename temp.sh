#!/bin/bash

# Source the conda.sh script
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate torchcell

# python
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py
python /Users/michaelvolk/Documents/projects/torchcell/experiments/003-fit-int/scripts/subset_data_module_load.py
wandb agent zhao-group/torchcell_003-fit-int-traditional_ml_dataset/ydfzzy73