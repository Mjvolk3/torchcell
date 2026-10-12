#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave9_bxctxprec_e200_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.wave9-isobutanol]]
#
# WAVE 9 BLOCK 2: the BETAXANTHIN is-the-module-on control, `bxctxprec`, seed 2700.
#
# WHAT IT ABLATES. The only ESTABLISHED betaxanthin contrast of wave 8 is that the precursor
# concentration readout beats 026's scalar flux head by +0.0399 (per seed +0.0377, +0.0532,
# +0.0288; sd 0.0123, n = 3, t = 5.6), while the scalar head is itself BELOW the no-module
# control by -0.0233. So the established statement is that the concentration readout recovers
# what the flux head lost, and nothing has ever measured whether the flux layer contributes
# to it.
#
# WHY IT MIGHT NOT. `FluxPrecursorConcentrationReadout` reads thermo_log_c, which is
# lo + (hi - lo) sigmoid(base + W context): a function of the pooled genotype context alone.
# The flux never enters the prediction; the layer touches the concentration only through the
# thermodynamic penalty that prices sign consistency between v and dG'(log c). This cell is
# the same head with no flux layer at all: the same zero base per GEM metabolite, the same
# Linear(90, 2806) from the perturbed-gene pool, the same window, the same five-precursor
# whole-cell mixture and the same affine. Absent by design: stoichiometry, kcat,
# thermodynamics, every constraint term, the flux vector.
#
# EVERY OTHER FLAG MATCHES gh_wave8_bxprec_e200_s2700.slurm, so the contrast is the layer
# alone.
#
# PRE-REGISTERED READ: paired by seed against wave8_bxprec (0.1404 / 0.1377 / 0.1159) and
# wave8_bxpool (0.1007 / 0.1159 / 0.1277). SUCCESS/KILL: |bxctxprec - bxprec| below 0.012,
# the prec-scalar sd, means the betaxanthin mechanism result is a parameterization result and
# the wave-8 conclusion has to be restated in the note. bxctxprec near bxpool means the layer
# really is doing the work on betaxanthin, which would be the first such evidence anywhere in
# the module.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave9_bxctxprec_e200_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w9-bxctxprec-s2700
#SBATCH --time=10:00:00
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
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE7_EPOCHS:-200}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm pooled --thermo-source union_external \
    --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --head betaxanthin --bx-readout context_precursor_concentration \
    --conc-space mixture --conc-upper-m 1.0 \
    --wandb-group gh_wave9_bx --tag _w9bxctxprec \
    --out wave9_bxctxprec_e200_s2700.json
