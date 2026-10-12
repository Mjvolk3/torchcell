#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --no-requeue
# experiments/036-dataset-fixes-before-kg-build/scripts/gh_genome_finding_rerun.slurm
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_genome_finding_rerun]]
#
# Read-only verification rerun for genome_finding_reclassification.py. Array task N
# verifies the dataset classes on line N of LANES_FILE through the pre-build sweep, from
# a frozen checkout (SNAPSHOT_ROOT, a detached worktree at the commit under test), with
# DATA_ROOT pointed at a per-task overlay (verification_overlay.py) so every report it
# writes lands in the overlay and nothing is written into the shared dev tree. The
# stores are read in place, read-only; none is rebuilt.
#
#   sbatch --array=1-<n> --time=<honest> --job-name=036-gf-rerun-<wave> \
#     --output=<dir>/%A_%a.out --error=<dir>/%A_%a.out \
#     --export=ALL,SNAPSHOT_ROOT=<dir>,LANES_FILE=<file>,RUN_DIR=<dir> \
#     experiments/036-dataset-fixes-before-kg-build/scripts/gh_genome_finding_rerun.slurm
echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell
: "${SNAPSHOT_ROOT:?set SNAPSHOT_ROOT, the frozen checkout to run}"
: "${LANES_FILE:?set LANES_FILE, one line of dataset classes per array task}"
: "${RUN_DIR:?set RUN_DIR, where overlays and lane results go}"
cd "$SNAPSHOT_ROOT" || exit 1
echo "SNAPSHOT_ROOT=$SNAPSHOT_ROOT ($(git -C "$SNAPSHOT_ROOT" rev-parse --short HEAD))"
export PYTHONPATH="$SNAPSHOT_ROOT"
set -a; source "$SNAPSHOT_ROOT/.env"; set +a
export WANDB_MODE=disabled
SOURCE_ROOT="$DATA_ROOT"
LANE="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OVERLAY="$RUN_DIR/overlays/$LANE"
mkdir -p "$RUN_DIR/overlays" "$RUN_DIR/lanes"
python experiments/036-dataset-fixes-before-kg-build/scripts/verification_overlay.py \
  --source "$SOURCE_ROOT" --overlay "$OVERLAY" || exit 1
# After .env: the overlay is the DATA_ROOT every route reads and writes under.
export DATA_ROOT="$OVERLAY"
CLASSES=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LANES_FILE")
ARGS=()
for cls in $CLASSES; do ARGS+=(--dataset "$cls"); done
echo "lane $LANE: $CLASSES"
python experiments/036-dataset-fixes-before-kg-build/scripts/kg4_pre_build_verification_sweep.py \
  "${ARGS[@]}" --out "$RUN_DIR/lanes/lane_${LANE}.json" || exit 1
echo "finished at $(date)"
