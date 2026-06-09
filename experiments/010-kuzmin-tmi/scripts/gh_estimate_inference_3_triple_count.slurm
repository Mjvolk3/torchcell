#!/bin/bash
#SBATCH -p main
#SBATCH --mem=128g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=010-est-inf3-triples
#SBATCH --time=8:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Estimate triple count for inference_dataset_3 with relaxed thresholds
# This loads SMF and DMF datasets and counts triples passing the new filters

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1

~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/estimate_inference_3_triple_count.py
