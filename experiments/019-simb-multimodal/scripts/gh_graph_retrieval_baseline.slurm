#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_graph_retrieval_baseline.slurm
# [[experiments.019-simb-multimodal.scripts.gh_graph_retrieval_baseline]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_graph_retrieval_baseline
#
# CPU-ONLY. B4, the cell graph as a retrieval key: a neighbor average over training strains
# retrieved by the deleted gene's adjacency row on each of the nine graphs and their union,
# against ProtT5, the own-profile oracle, a random embedding and degree-preserving rewired
# controls, on the v13 expression and v14 proteome partitions (split seeds 0 to 3), with the
# CGT prediction dumps scored under the same rule. One task, everything in one process.
#
#   sbatch experiments/019-simb-multimodal/scripts/gh_graph_retrieval_baseline.slurm
#
# No --gres=gpu on purpose. The cost is nine configuration-model rewirings (the
# coexpression graph carries ~1M edges) and, per split seed and key, a [n_eval, n_train]
# cosine matrix over 6,607-wide rows: minutes. Keys are built one at a time, so the peak
# is one [3,581, 6,607] float32 key matrix plus the LMDB matrices; the 32g is slack.
#
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=019-b4-graph-retrieval
#SBATCH --time=04:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out

set -euo pipefail

mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

# The checkout whose script was submitted, so the code that runs is the code that was
# committed (SLURM_SUBMIT_DIR is where sbatch was invoked; BASH_SOURCE is /var under slurm).
if [ -n "${B4_PROJECT_ROOT:-}" ]; then
  PROJECT_ROOT="$B4_PROJECT_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/torchcell" ]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  echo "FATAL: submit from a torchcell checkout or set B4_PROJECT_ROOT" >&2
  exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

# BLAS thread count pinned to the allocation, so numpy does not spawn one thread per
# physical core regardless of what SLURM granted.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
export ASSET_IMAGES_DIR="${ASSET_IMAGES_DIR:-/home/michaelvolk/Documents/projects/torchcell/notes/assets/images}"

# Optional: B4_ARGS carries extra flags (a seed subset, --report-md <path>, ...).
read -r -a EXTRA <<< "${B4_ARGS:-}"

echo "=========================================================="
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-none}"
echo "root    : $PROJECT_ROOT"
echo "source  : $(git -C "$PROJECT_ROOT" rev-parse --short HEAD) + working tree"
echo "args    : ${EXTRA[*]:-}"
echo "=========================================================="

cd "$PROJECT_ROOT"
exec python experiments/019-simb-multimodal/scripts/graph_retrieval_baseline.py "${EXTRA[@]}"
