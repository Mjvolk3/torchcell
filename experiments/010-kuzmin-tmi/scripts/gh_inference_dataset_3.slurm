#!/bin/bash
#SBATCH -p main
#SBATCH --mem=500g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=010-inf-dataset-3
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Create LMDB dataset for inference_3 (parquet -> LMDB)
# Expects triple_combinations_list.parquet from generate_triple_combinations_inference_3.py

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Set number of workers for parallel processing
export NUM_WORKERS=64

~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/inference_dataset_3.py
