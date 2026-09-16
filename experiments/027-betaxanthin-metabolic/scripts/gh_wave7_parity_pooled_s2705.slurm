#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave7_parity_pooled_s2705.slurm
# [[experiments.027-betaxanthin-metabolic.wave7-parity-one-table]]
#
# WAVE 7 betaxanthin parity, seed 2705: the pooled control, no metabolic module, which is what the arm under test is measured against.
#
# WHY WAVE 7 EXISTS. Wave 6's six parity seeds resolved THREE different kcat tables
# (148/0/3410/170 at seeds 2700, 2701 and 2703; 96/782/2820/30 at 2702; 148/0/3570/10 at
# 2704 and 2705), because the resolver reads the predicted-parameter mirror off $DATA_ROOT
# and that mirror was regenerated while the queue drained. None of the six is the shipped
# verified table. Wave 7 reruns the same recipe at the same six seeds with every cell on
# ONE table, the shipped 97 measured / 782 homolog / 2831 predicted / 18 default, verified
# before submission by verify_kcat_table.py and checked again in every result file's
# flux_meta.kcat_tier_counts.
#
# WHAT CHANGED FROM THE WAVE-6 SCRIPT, and it is exactly one thing. The wave-6 cells pinned
# PYTHONPATH and TC_FLUX_SCRIPTS_DIR at the 23:05 2026-09-10 freeze, whose train_flux.py
# predates the kcat_homolog_tier switch and therefore CANNOT build the shipped table. Wave 7
# drops that pin and runs the code in the job snapshot freeze_and_submit.sh makes, which is
# the current worktree. Both arms of every pair run from a snapshot of the same code state,
# and train_bx.py records the sha256 of the train_bx.py and flux_layer.py it imported into
# each result file, so a mid-run edit is detectable rather than silent.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave7_parity_pooled_s2705.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w7-pooled-s2705
#SBATCH --time=05:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TC_FLUX_SCRIPTS_DIR="$PROJECT_ROOT/experiments/026-metabolism-flux/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/027-betaxanthin-metabolic/wandb
export WANDB_PROJECT=torchcell_027_bxfx
export NUM_WORKERS=2
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave7
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE7_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm pooled --seed 2705 --epochs "$EPOCHS" --crps --graph-reg \
    --head betaxanthin \
    --wandb-group gh_wave7_000 --tag _w7 \
    --out wave7_parity_pooled_s2705.json
