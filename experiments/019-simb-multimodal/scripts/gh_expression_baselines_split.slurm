#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_expression_baselines_split.slurm
# [[experiments.019-simb-multimodal.scripts.gh_expression_baselines_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_expression_baselines_split
#
# CPU-ONLY. The Ahlmann-Eltze baselines (B0 mean, B1 no change, B2 bilinear ridge, B3
# embedding neighbors) on the SAME partitions the v13 split round trains on. Five tasks:
#
#   task 0-3   split seeds 0-3, 80/10/10 (val-selected, test reported)
#   task 4     split seed 0 with test folded into train (the 90/10 arm, val only)
#
#   sbatch experiments/019-simb-multimodal/scripts/gh_expression_baselines_split.slurm
#
# No --gres=gpu on purpose; the GPU runs are on IGB. Each task is a ridge sweep over a
# [1244, 6169] matrix and four embeddings: minutes, and the 32g is slack.
#
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=019-baselines-split
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail

mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

# The checkout whose script was submitted, so the code that runs is the code that was
# committed (SLURM_SUBMIT_DIR is where sbatch was invoked; BASH_SOURCE is /var under slurm).
if [ -n "${BASELINES_PROJECT_ROOT:-}" ]; then
  PROJECT_ROOT="$BASELINES_PROJECT_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/torchcell" ]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  echo "FATAL: submit from a torchcell checkout or set BASELINES_PROJECT_ROOT" >&2
  exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"
case "$TASK" in
  0|1|2|3) ARGS=(--split-seed "$TASK") ;;
  4)       ARGS=(--split-seed 0 --fold-test-into-train) ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac

echo "=========================================================="
echo "host    : $(hostname)"
echo "job     : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK}"
echo "root    : $PROJECT_ROOT"
echo "source  : $(git -C "$PROJECT_ROOT" rev-parse --short HEAD) + working tree"
echo "args    : ${ARGS[*]}"
echo "=========================================================="

cd "$PROJECT_ROOT"
exec python experiments/019-simb-multimodal/scripts/expression_baselines_split.py "${ARGS[@]}"
