#!/bin/bash
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=010-inf4-generate
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Writes the inference_4 triple LMDB. CPU only: the cost is JSON serialization and
# roughly 163 GB of sequential LMDB writes, no GPU involved.

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/inference_4_generate_triples.py \
  --min-screens 50 \
  --min-supported 1
