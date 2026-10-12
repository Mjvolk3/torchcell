#!/bin/bash
#SBATCH --job-name=1045-verify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/036-dataset-fixes-before-kg-build/scripts/gh_1045_store_verify.slurm
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_1045_store_verify]]
#
# Issue #1045 (the removals audit): verify ONE rebuilt store through the pre-build
# sweep, read-only on processed/, into its own lane results file. The Costanzo 2016 and
# Kuzmin 2020 stores have no per-dataset verifier, so `build_dataset_lmdb --verify`
# reports none for them and their family runner is run here. TORCHCELL_SRC is a frozen
# checkout of the branch (a `git clone --shared` at the commit), so editing the worktree
# while this runs cannot change the verifier it imports. Sizing: dmf_costanzo2016 took
# 3 h 31 min at 2 CPUs / 32 G under #889 (job 3766).
#
#   sbatch --export=ALL,TORCHCELL_SRC=<checkout>,DATASET_CLASS=DmfCostanzo2016Dataset,\
#     LANE_OUT=<lane.json> \
#     experiments/036-dataset-fixes-before-kg-build/scripts/gh_1045_store_verify.slurm
set -euo pipefail
echo "Running on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
: "${TORCHCELL_SRC:?set TORCHCELL_SRC, the frozen checkout to verify with}"
: "${DATASET_CLASS:?set DATASET_CLASS, e.g. DmfCostanzo2016Dataset}"
# A lane file per job: two jobs writing the shared results JSON would race.
: "${LANE_OUT:?set LANE_OUT, the lane results file this job writes}"
cd "$TORCHCELL_SRC"
echo "source: $TORCHCELL_SRC ($(git rev-parse --short HEAD))"
set -a
source "$TORCHCELL_SRC/.env"
set +a
export WANDB_MODE=disabled
PYTHONPATH="$TORCHCELL_SRC" python \
  experiments/036-dataset-fixes-before-kg-build/scripts/kg4_pre_build_verification_sweep.py \
  --dataset "$DATASET_CLASS" --out "$LANE_OUT"
echo "finished at $(date)"
