#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_compare_raw_stores_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_compare_raw_stores_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_compare_raw_stores_043.slurm
#
# Record-level, order-free equivalence of raw_043_v2 (single session, job 6187) and
# raw_043_v2_partitioned (job 6208): compare_raw_stores_043.py. Submit only once both raw
# stages have closed (raw/gene_set.json written in both). Submit from the repository root:
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_compare_raw_stores_043.slurm
#
# Memory: 128 GB; each raw/experiment_reference_index.json is about 2.6 GB of JSON and both
# are held parsed at once. No GPU.
#SBATCH -p main
#SBATCH --mem=128g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=043-compare-raw
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/043-expression-proteome-morphology/scripts/compare_raw_stores_043.py --workers 16
echo "done $(date)"
