#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_cross_study_conditioning_oracle.slurm
# [[experiments.019-simb-multimodal.scripts.gh_cross_study_conditioning_oracle]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_cross_study_conditioning_oracle
#
# CPU-ONLY. Separates biology from same-array measurement structure in the masked
# conditioning oracle, using Kemmeren 2014 vs Sameith 2015 on their 82 shared deletions.
#
#   sbatch --export=ALL,ORACLE_PROJECT_ROOT=$PWD \
#       experiments/019-simb-multimodal/scripts/gh_cross_study_conditioning_oracle.slurm
#
# NO --gres=gpu ON PURPOSE. This is a covariance solve in numpy; a GPU request would idle
# a card that the training wave needs.
#
# Sizing: the full [F, F] covariance is never formed -- only the Sigma_MM and Sigma_UM
# blocks -- so peak memory is the two [~1500, ~6000] float64 study matrices (~150 MB) plus
# a [5000, 1000] block. The 32g request is slack, not need. Cost is dominated by the
# Sigma_UM product at m = 1000, re-formed once per (draw, arm, lam): ~38 solves per draw,
# 5 draws, i.e. minutes on 8 threads. 04:00:00 is generous headroom.
#
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=019-xstudy-oracle
#SBATCH --time=04:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out

set -euo pipefail

mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

PROJECT_ROOT="${ORACLE_PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/019-decoder-pair-routing}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

# BLAS thread count pinned to the allocation. numpy would otherwise spawn one thread per
# physical core regardless of what SLURM granted, oversubscribing the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PROJECT_ROOT"

echo "=========================================================="
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-none}"
echo "root    : $PROJECT_ROOT"
echo "threads : $OMP_NUM_THREADS"
echo "=========================================================="

cd "$PROJECT_ROOT"
exec python experiments/019-simb-multimodal/scripts/cross_study_conditioning_oracle.py
