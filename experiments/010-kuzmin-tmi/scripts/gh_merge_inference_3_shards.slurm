#!/bin/bash
#SBATCH -p main
#SBATCH --mem=128g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=010-merge-inf3
#SBATCH --time=4:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# INFERENCE_3: Merge per-rank parquet shards into single sorted file
# Casts string -> large_string to avoid 2GB offset overflow on ~465M rows
# No GPU needed — CPU + memory only

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_inference_3.py \
  --merge-only
