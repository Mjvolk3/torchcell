#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_masked_conditioning_oracle.slurm
# [[experiments.019-simb-multimodal.scripts.gh_masked_conditioning_oracle]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_masked_conditioning_oracle
#
# CPU-ONLY. Bounds what a masked-label objective can buy, before any of it is built.
#
#   sbatch experiments/019-simb-multimodal/scripts/gh_masked_conditioning_oracle.slurm
#
# NO --gres=gpu ON PURPOSE. This is a covariance solve, not a training run, and the four
# GPUs are reserved for the wave. Requesting one would idle it for the duration.
#
# Sizing: the full [F, F] covariance is never formed -- only the Sigma_UM and Sigma_MM
# blocks, so peak memory is tens of MB plus the [1482, 6169] float32 matrix (~37 MB). The
# 32g request is slack, not need. The dominant cost is N_DRAWS x |RIDGE_GRID| solves of a
# [m, m] system at m = 1000, i.e. 30 solves of a 1000x1000 -- seconds, not minutes.
#
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=019-mask-oracle
#SBATCH --time=02:00:00
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
echo "source  : $(git -C "$PROJECT_ROOT" rev-parse --short HEAD) + working tree"
echo "threads : $OMP_NUM_THREADS"
echo "=========================================================="

cd "$PROJECT_ROOT"
exec python experiments/019-simb-multimodal/scripts/masked_conditioning_oracle.py
