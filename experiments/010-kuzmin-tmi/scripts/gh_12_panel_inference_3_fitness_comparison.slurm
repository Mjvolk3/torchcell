#!/bin/bash
#SBATCH -p main
#SBATCH --mem=4g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=010-fitness-compare-inf3
#SBATCH --time=0:10:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# INFERENCE_3: Fitness measurement comparison across sources (Panel 12)
# Reads queried CSVs, produces forest plots, Gaussian overlays, summary CSV
# Lightweight: no dataset loading, no GPU — just CSV + matplotlib

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/12_panel_inference_3_fitness_comparison.py
