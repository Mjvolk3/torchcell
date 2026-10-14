#!/bin/bash
# experiments/025-solid-growth/scripts/gh_graph_reg_flags_smoke.slurm
# [[experiments.025-solid-growth.scripts.graph_reg_flags_smoke]]
#
# CPU smoke of the round-2 graph-entry flags at real size (6,607 genes, eight layers): one
# forward and backward per variant, no LMDB, no GPU. Runs from the submitting worktree.
#
#   sbatch experiments/025-solid-growth/scripts/gh_graph_reg_flags_smoke.slurm
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=025-flags-smoke
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
PY="$HOME/miniconda3/envs/torchcell/bin/python"
echo "job $SLURM_JOB_ID project $PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null)) $(date)"
cd "$PROJECT_ROOT" || exit 1
set -a; source "$PROJECT_ROOT/.env"; set +a
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16
"$PY" - <<PY || exit 1
import torchcell
assert torchcell.__file__.startswith("$PROJECT_ROOT"), torchcell.__file__
print("torchcell:", torchcell.__file__)
PY
"$PY" experiments/025-solid-growth/scripts/graph_reg_flags_smoke.py
echo "finished $(date)"
