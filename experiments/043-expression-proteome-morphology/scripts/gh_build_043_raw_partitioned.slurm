#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_build_043_raw_partitioned.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_build_043_raw_partitioned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_build_043_raw_partitioned.slurm
#
# The partitioned CANDIDATE of the store v2 raw stage (build_043_raw_partitioned.py),
# beside the single-session reference (job 6187). Raw stage only, into
# raw_043_v2_partitioned; compare_raw_stores_043.py then proves record equivalence.
# Submit from the repository root:
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_build_043_raw_partitioned.slurm
#
# Memory: 64 GB. In flight are PARTITIONS_IN_FLIGHT_PER_WORKER (2) x 8 workers = 16
# partitions; the largest block (Boocock 2025, 79,516 records) gives about 311 records per
# partition at prefix length 2, about 0.47 GB of stored JSON at the store average of about
# 1.5 MB per non-Vaishnav record, so about 7.5 GB queued in the writer. Hypothesis
# (untested): each worker's transient decode of one partition stays under about 5 GB, so
# 8 workers add about 40 GB. No GPU.
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=043-raw-partitioned
#SBATCH --time=08:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export STORE_043_VERSION=v2
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/043-expression-proteome-morphology/scripts/build_043_raw_partitioned.py --fetch-workers 8 --prefix-length 2
echo "done $(date)"
