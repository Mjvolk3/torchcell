#!/bin/bash
#SBATCH -p main
#SBATCH --mem=500g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=010-gen-triples-inf3
#SBATCH --time=8:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Generate triple combinations for inference_dataset_3 with relaxed thresholds
# Thresholds: max(smf) > 1.04, all(smf) > 0.90, max(dmf) > 1.08, all(dmf) > 0.90

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/generate_triple_combinations_inference_3.py
