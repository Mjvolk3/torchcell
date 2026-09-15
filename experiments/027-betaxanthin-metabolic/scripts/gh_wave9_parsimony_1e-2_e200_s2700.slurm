#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave9_parsimony_1e-2_e200_s2700.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# WAVE 9, E4: PARSIMONY, PRICED OUTSIDE THE RAMP FOR THE FIRST TIME.
#
# THE OMISSION. `constraint_loss` multiplies the whole physics sum, `c_parsimony` included,
# by `constraint_scale`, and every wave-7 and wave-8 cell runs that scale at exactly 0.0. So
# the parsimony weight those cells record, 1e-3, has been multiplied by zero in every one of
# them: parsimony has never been applied in this experiment at all. That matters because
# parsimony is what makes an FBA flux sparse. The pFBA reference carries flux on 10.3 percent
# of reactions (experiments/026-metabolism-flux/results/probe_fba_rule.json) against the
# boxforce head's 28 to 32 percent, and the density term, which WAS applied, pushes that
# fraction the wrong way: it was built to raise the active-row count from 422 when the
# reference is 83.
#
# `parsimony_outside_ramp_weight` adds `c_parsimony` outside the product, the same way
# `density_weight` and `balance_active_weight` are already added, and for the same reason.
#
# THE ARM. fluxforce_k100_boxforce at density 0.0 and active-row balance 0.0, which is the
# weight-zero control of jobs 1767/1770/1771, plus the parsimony weight. Seed 2700, 200
# epochs. The sibling cell runs 1e-3.
#
# PRE-REGISTERED READ (review, rank 4): `flux_active_frac` must FALL toward 0.107 and
# `balance_active_rows_1e-03` toward 83; `bal_act_p95` is the outcome.
# KILL: if `flux_active_frac` does not fall below 0.20 at the strong weight, the head cannot
# be made sparse by a penalty and the box parameterization is the blocker, which is what
# results/box_zero_reachability.json already predicts for a sigmoid magnitude.
#
# ----------------------------------------------------------------------------------------
# BLOCKED ON A ONE-FLAG PATCH TO train_bx.py, WHICH THIS SESSION DOES NOT OWN.
#
# `--flux-parsimony-weight` does not exist yet. The layer side is landed
# (FluxLayerConfig.parsimony_outside_ramp_weight, read by constraint_loss at call time, 0.0
# by default so every earlier cell is unchanged), and the trainer side is five lines in
# train_bx.py, written out verbatim in
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]. This cell will die with
# "unrecognized arguments" until that patch is applied.
# ----------------------------------------------------------------------------------------
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave9_parsimony_1e-2_e200_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w9-parsimony-1e-2-s2700
#SBATCH --time=08:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/027-betaxanthin-metabolic/wandb
export WANDB_PROJECT=torchcell_027_bxfx
export NUM_WORKERS=2
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave7
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; parsimony outside the ramp"
EPOCHS="${WAVE9_EPOCHS:-200}"
SEED="${WAVE9_SEED:-2700}"
PARSIMONY="${WAVE9_PARSIMONY:-1e-2}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_fluxforce_boxforce --flux-density-weight 0.0 --flux-balance-active-weight 0.0 \
    --flux-parsimony-weight "$PARSIMONY" \
    --seed "$SEED" --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave9_parsimony --tag "parsimony${PARSIMONY}" \
    --out "wave9_parsimony_${PARSIMONY}_boxforce_k100_e${EPOCHS}_s${SEED}.json"
