#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_lowrank_output_ceiling.slurm
# [[experiments.019-simb-multimodal.scripts.gh_lowrank_output_ceiling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_lowrank_output_ceiling
#
# CPU-ONLY. Measures the ceiling of a rank-r OUTPUT head before any rank is run on GPU.
#
#   sbatch --export=ALL,ORACLE_PROJECT_ROOT=$PWD \
#     experiments/019-simb-multimodal/scripts/gh_lowrank_output_ceiling.slurm
#
# NO --gres=gpu ON PURPOSE. This is three SVDs in numpy, not a training run, and the GPUs
# are reserved for the wave. Requesting one would idle it for the duration.
#
# Sizing: the dominant cost is the thin SVD of the [1327, 6169] float64 fit residual (~65 MB
# in, U/Vt ~80 MB out), plus a second on the [1172, 6169] train-only residual and a third on
# the [155, 6169] val residual. Minutes at most on 8 threads; the 32g request is slack, not
# need. The 6169^2 covariance is never formed.
#
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=019-lowrank-ceiling
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
# physical core regardless of what SLURM granted, oversubscribing the node -- and this job
# is almost entirely LAPACK gesdd, so it would take the whole machine.
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
exec python experiments/019-simb-multimodal/scripts/lowrank_output_ceiling.py
