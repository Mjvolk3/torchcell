#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_build_043_store.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_build_043_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_build_043_store.slurm
#
# Build the experiment 043 raw store (15 expression, proteome and morphology datasets of
# KG 4.0, one sample per served experiment) from the served graph on GilaHyper, under
# slurm (DB work never runs as a login-shell process). Submit from the repository root:
#
#   sbatch experiments/043-expression-proteome-morphology/scripts/gh_build_043_store.slurm
#
# Memory: 256 GB. Job 5855 (96 GB) finished the raw stage (173,617 records, 163 GB raw
# LMDB, 5 h 48 min) and was OOM-killed at 61% of the aggregation pass: the aggregation
# LMDB was still 8 KB, so the stage's single write transaction held every copied record
# in memory (MaxRSS 100.4 GB). The aggregation and processed copies each write the whole
# store in one transaction. A rerun reuses the finished raw LMDB (the query is skipped
# when raw/lmdb/data.mdb exists).
#SBATCH -p main
#SBATCH --mem=256g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=043-build-store
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/043-expression-proteome-morphology/scripts/build_043_store.py
echo "done $(date)"
