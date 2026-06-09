#!/bin/bash
#SBATCH -p main
#SBATCH --mem=8g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=010-inf1-vs-inf2
#SBATCH --time=0:10:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Change to the project root directory
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

# Add project to PYTHONPATH
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Compare inference_1 vs inference_2 gene panels
~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/inference_dataset_1_vs_2.py
