#!/bin/bash
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=010-inf4-measured
#SBATCH --time=04:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Score the 4,877 already-measured inference_4 triples against their 010 labels.
#
# NO GPU. There is deliberately no --gres line: the predictions for the whole
# 41,877,232-triple space already exist as Parquet, so this is a join plus an LMDB
# pass. Requesting a GPU would take one away from the training run for no reason.
#
# Memory: the three checkpoint prediction vectors are 41.9M float32 each, plus the
# argsort that assigns every triple its rank, so 96 GB is comfortable headroom.

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

# Keep BLAS off the training job's cores; the work here is memory-bound anyway.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python -u experiments/010-kuzmin-tmi/scripts/inference_4_measured_diagnostic.py
