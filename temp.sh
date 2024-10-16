#!/bin/bash

# Source the conda.sh script
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate torchcell

# python
# python /Users/michaelvolk/Documents/projects/torchcell/torchcell/data/neo4j_cell.py
python /Users/michaelvolk/Documents/projects/torchcell/experiments/003-fit-int/scripts/create_cached_rename_modules.py
wandb agent zhao-group/torchcell_003-fit-int-traditional_ml_dataset/zq2wns4x