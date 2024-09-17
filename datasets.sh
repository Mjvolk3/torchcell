#!/bin/bash

# Source the conda.sh script
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate torchcell

# Run the Python scripts
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/sgd.py
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/synth_leth_db.py
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/kuzmin2020.py
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/kuzmin2018.py
python /Users/michaelvolk/Documents/projects/torchcell/torchcell/datasets/scerevisiae/costanzo2016.py
