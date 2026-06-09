#!/bin/bash
#SBATCH -p main
#SBATCH --mem=500g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=010-gen-triples-inf2
#SBATCH --time=4:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Change to the project root directory (required for relative paths in Python script)
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Activate conda environment directly
source ~/miniconda3/bin/activate
conda activate torchcell

# Add project to PYTHONPATH
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Run triple generation for inference_dataset_2
~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/generate_triple_combinations_inference_2.py
