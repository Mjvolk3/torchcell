#!/bin/bash
#SBATCH --job-name=036-889-verify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat-verification-uncovered-stores-889/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat-verification-uncovered-stores-889/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/036-dataset-fixes-before-kg-build/scripts/gh_no_verifier_store_sweep.slurm
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_no_verifier_store_sweep]]
#
# Issue #889: verify ONE of the two 20,705,612-record Costanzo 2016 stores through the
# pre-build sweep, read-only on processed/, into its own lane results file (merged into
# results/kg4_pre_build_verification_sweep.json with --merge afterwards). The other 14
# #889 stores are small enough to run in a shell lane.
#
#   sbatch --export=ALL,DATASET_CLASS=DmfCostanzo2016Dataset,LANE_OUT=<lane.json> \
#     experiments/036-dataset-fixes-before-kg-build/scripts/gh_no_verifier_store_sweep.slurm
echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell
PROJECT_ROOT="/home/michaelvolk/Documents/projects/torchcell.worktrees/feat-verification-uncovered-stores-889"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"
set -a; source "$PROJECT_ROOT/.env"; set +a
export WANDB_MODE=disabled
: "${DATASET_CLASS:?set DATASET_CLASS, e.g. DmfCostanzo2016Dataset}"
# A lane file per job: two jobs writing the shared results JSON would race.
: "${LANE_OUT:?set LANE_OUT, the lane results file this job writes}"
python experiments/036-dataset-fixes-before-kg-build/scripts/kg4_pre_build_verification_sweep.py \
  --dataset "$DATASET_CLASS" --out "$LANE_OUT" || exit 1
echo "finished at $(date)"
