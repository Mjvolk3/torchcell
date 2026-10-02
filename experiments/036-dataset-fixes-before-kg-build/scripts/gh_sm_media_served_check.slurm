#!/bin/bash
#SBATCH --job-name=036-sm-media-check
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/143-sm-media-served-check/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/143-sm-media-served-check/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/036-dataset-fixes-before-kg-build/scripts/gh_sm_media_served_check.slurm
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_sm_media_served_check]]
#
# Media-node census of the served graph for the issue #143 close-out
# (sm_media_served_check.py). Anything that touches the served store runs under slurm.
#
#   sbatch experiments/036-dataset-fixes-before-kg-build/scripts/gh_sm_media_served_check.slurm
echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell
PROJECT_ROOT="/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/143-sm-media-served-check"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"
set -a; source "$PROJECT_ROOT/.env"; set +a
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="torchcell"
export WANDB_MODE=disabled
python experiments/036-dataset-fixes-before-kg-build/scripts/sm_media_served_check.py \
  --out experiments/036-dataset-fixes-before-kg-build/results/sm_media_served_2026-10-02.csv || exit 1
echo "finished at $(date)"
