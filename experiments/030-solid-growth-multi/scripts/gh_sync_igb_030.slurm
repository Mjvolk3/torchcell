#!/bin/bash
# experiments/030-solid-growth-multi/scripts/gh_sync_igb_030.slurm
# [[experiments.030-solid-growth-multi.scripts.gh_sync_igb_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/gh_sync_igb_030.slurm
#
# WHY. Run the 030 build mirror (sync_igb_030_build.sh, GilaHyper -> IGB login node) under
# slurm so the 2.5 to 3.5 h transfer survives the session that started it. rsync is
# resumable, so a killed or timed-out job is resubmitted, not repaired. CPU only: the job
# is one rsync over ssh, so 4 CPUs and 8 GB are plenty (same as gh_sync_igb_029.slurm).
#
# The output directory must exist before sbatch, since slurm opens --output first:
#   mkdir -p experiments/030-solid-growth-multi/slurm/output
#
# RUN FROM the worktree root on GilaHyper:
#   sbatch experiments/030-solid-growth-multi/scripts/gh_sync_igb_030.slurm
#   sbatch --export=ALL,PROJECT_ROOT=<other worktree> experiments/030-solid-growth-multi/scripts/gh_sync_igb_030.slurm
#
# Rerun after the GilaHyper smoke job writes data_module_cache/; the second pass sends
# only the cache and the launcher preflight on IGB then passes.
#SBATCH --job-name=030-sync-igb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --no-requeue

set -euo pipefail
echo "Running on $(hostname) at $(date)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token}"
cd "$PROJECT_ROOT"
# conda.sh, not ~/.zshrc: the zsh rc file breaks under set -e and this shell is bash.
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
# shellcheck disable=SC1091
set -a; source "$PROJECT_ROOT/.env"; set +a
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
bash experiments/030-solid-growth-multi/scripts/sync_igb_030_build.sh
echo "finished at $(date)"
