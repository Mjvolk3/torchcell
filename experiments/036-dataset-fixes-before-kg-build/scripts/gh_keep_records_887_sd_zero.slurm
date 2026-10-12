#!/bin/bash
#SBATCH --job-name=036-887-sd0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/fix/keep-records-887-sd-zero/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/fix/keep-records-887-sd-zero/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/036-dataset-fixes-before-kg-build/scripts/gh_keep_records_887_sd_zero.slurm
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_keep_records_887_sd_zero]]
#
# #887 and the released SDs of 0 (#894, #895, #897, #930): measure the pinned releases
# and the dev stores (keep_records_887_sd_zero_inventory.py), then re-run the L0-L4
# verification of the four stores those issues name, read-only on processed/, into a
# lane results file. No store is built or rebuilt.
#
#   sbatch experiments/036-dataset-fixes-before-kg-build/scripts/gh_keep_records_887_sd_zero.slurm
echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell
PROJECT_ROOT="/home/michaelvolk/Documents/projects/torchcell.worktrees/fix/keep-records-887-sd-zero"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"
set -a; source "$PROJECT_ROOT/.env"; set +a
export WANDB_MODE=disabled
SCRIPTS=experiments/036-dataset-fixes-before-kg-build/scripts
python "$SCRIPTS/keep_records_887_sd_zero_inventory.py" || exit 1
LANE=experiments/036-dataset-fixes-before-kg-build/results/keep_records_887_sd_zero_verification.json
for DATASET in FattyAcidXue2025Dataset DmfKuzmin2018Dataset TmfKuzmin2018Dataset DmfKuzmin2020Dataset; do
  echo "verify $DATASET at $(date)"
  python "$SCRIPTS/kg4_pre_build_verification_sweep.py" --dataset "$DATASET" --out "$LANE" || exit 1
done
echo "finished at $(date)"
