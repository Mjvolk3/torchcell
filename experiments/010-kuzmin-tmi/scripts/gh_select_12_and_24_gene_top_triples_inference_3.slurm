#!/bin/bash
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=010-select-genes-inf3
#SBATCH --time=2:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# INFERENCE_3: Gene panel selection from inference results
# Thresholds: max(SMF) > 1.04, max(DMF) > 1.08
# Designed for JT test validation

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Run gene selection analysis on inference_3 results
~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/select_12_and_24_genes_top_triples_inference_3.py
