#!/bin/bash
#SBATCH --job-name=036-887-rebuild
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/fix/keep-records-887-sd-zero/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/fix/keep-records-887-sd-zero/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/036-dataset-fixes-before-kg-build/scripts/gh_keep_records_887_sd_zero_rebuild.slurm
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_keep_records_887_sd_zero_rebuild]]
#
# PR #1054: rebuild, in the DEV tree, the two stores whose records this branch changes
# (Kuzmin 2018 Smf and Dmf: the query strains' Additional Data S4 SDs), each with
# --retire-existing --verify, then census them. tmf_kuzmin2020 (its SD now labeled) is
# NOT rebuilt here: fix/audit-yeast-refusals-1045 rebuilt every Kuzmin 2020 store with
# the retired and merged ORF rows it keeps, which this branch's tree would drop again;
# the handover is on issue #1045.
#
#   sbatch experiments/036-dataset-fixes-before-kg-build/scripts/gh_keep_records_887_sd_zero_rebuild.slurm
set -euo pipefail
echo "Running on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
PROJECT_ROOT="/home/michaelvolk/Documents/projects/torchcell.worktrees/fix/keep-records-887-sd-zero"
cd "$PROJECT_ROOT"
echo "source: $PROJECT_ROOT ($(git rev-parse --short HEAD))"
export PYTHONPATH="$PROJECT_ROOT"
set -a; source "$PROJECT_ROOT/.env"; set +a
export WANDB_MODE=disabled
for DATASET in SmfKuzmin2018Dataset DmfKuzmin2018Dataset; do
  echo "build $DATASET at $(date)"
  python -m torchcell.database.build_dataset_lmdb --dataset "$DATASET" \
    --io-workers "$SLURM_CPUS_PER_TASK" --retire-existing --verify
done
python experiments/036-dataset-fixes-before-kg-build/scripts/keep_records_887_sd_zero_inventory.py --after
echo "DONE at $(date)"
