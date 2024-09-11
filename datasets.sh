#!/bin/bash

# Source the conda.sh script
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate torchcell

# Run the Python scripts
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/sgd_gene_essentiality.py
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/syn_leth_db_yeast.py
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/kuzmin2020.py
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/kuzmin2018.py
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/costanzo2016.py