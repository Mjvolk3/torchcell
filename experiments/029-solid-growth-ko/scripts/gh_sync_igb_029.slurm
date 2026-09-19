#!/bin/bash
#SBATCH --job-name=029-sync-igb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/025-fitness-joint-head/experiments/029-solid-growth-ko/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/025-fitness-joint-head/experiments/029-solid-growth-ko/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Run the 029 processed-LMDB mirror to IGB under slurm so it survives the session that
# started it. rsync is resumable, so a killed job is resubmitted, not repaired.
set -euo pipefail
echo "Running on $(hostname) at $(date)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/025-fitness-joint-head}"
cd "$PROJECT_ROOT"
set -a; source "$PROJECT_ROOT/.env"; set +a
bash experiments/029-solid-growth-ko/scripts/sync_igb_029_build.sh
echo "finished at $(date)"
