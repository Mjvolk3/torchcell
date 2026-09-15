#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave9b_ibprec_e200_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.wave9-isobutanol]]
#
# WAVE 9b: the isobutanol three-way re-run, `ibprec`, seed 2700.
#
# THIS CELL IS the concentration readout, now on THREE anchors.
#
# WHY THERE IS A WAVE 9b. The seed-2700 pair of wave 9 ran to 200 epochs and neither arm is
# usable as it stands (jobs 1926 and 1928, W&B 6z2ovude and s4z0ur7o):
#
# * `ibpool` DID learn. Its validation prediction spread left the mean plateau by epoch 10
#   (0.0055), reached 0.096 by epoch 16, and its selected epoch 21 scored pinned Spearman
#   0.1278 on the 504 held-out genes, which is inside the wave-8 betaxanthin pooled range
#   (0.1007 / 0.1159 / 0.1277). What followed was 175 epochs of OVERFITTING: train data loss
#   0.553 -> 0.371 while validation loss rose 0.554 -> 0.712 and validation Pearson went
#   negative. The end-of-run numbers describe the overfit tail, not the selected epoch.
# * `ibprec` never learned anything. Its train data loss sat between 0.538 and 0.548 for 195
#   consecutive epochs, its validation prediction spread never exceeded 0.0022 and read
#   0.00009 at epoch 199, and its selected epoch 13 was picked off a validation curve whose
#   own noise floor is larger than the value selected.
#
# THREE CANDIDATE CAUSES WERE CHECKED AND REFUTED, so they are not what 9b changes.
#
# 1. Target scale. The target IS standardized on the train pool: mean 0.0124 and sd 0.1831 in
#    log10 fold change, recorded in both checkpoints, and the loss is charged on
#    (t - mean) / sd, so the CRPS sigma head sees a unit-variance target. Both arms used the
#    SAME mean and sd and one of them escaped, so the scale cannot be the discriminator.
# 2. Loss masking. The train split holds 9,733 records of which 2,981 carry a supervised
#    isobutanol target, 30.6 percent, about 39 per 128-row batch. Thin against betaxanthin's
#    95 percent, but not starvation.
# 3. Constraint balance. The constraint-to-data loss ratio is 0.05 for `ibprec` at epoch 20,
#    100 and 199, and it is ALSO 0.05 at all three for wave-8 `bxprec`, which escaped.
#
# WHAT DID SEPARATE THEM, measured. Wave-8 `bxprec` is the same readout class on the same
# layer and window, and it escaped: spread 0.0002 at epoch 20, 0.023 at epoch 40, 0.066 at
# epoch 60, 0.352 at epoch 80, selected epoch 109, data loss 0.528 -> 0.421. Against that,
# `ibprec` is not a slow escaper, it is flat. Its concentration-head weight row at the anchor
# is 0.844 against a global row-norm median of 0.583, so the head is not dead; the pool it
# reads is not moving. The two structural differences between the arms are the ANCHOR COUNT,
# 1 against 5, and the reaction-energy table, `verified` against `union_external`.
#
# Hypothesis (untested): a single anchor is too narrow a channel, because the whole
# genotype-to-phenotype map has to pass through one scalar that is simultaneously held by the
# second law and the dissipation limit across 2,806 metabolites.
#
# WHAT 9b CHANGES, and it is three flags shared by all three arms plus one on `ibprec`:
#
# * `--thermo-source union_external` everywhere, matching the wave-8 betaxanthin three-way
#   exactly, so the one precursor arm known to escape and this one differ by target alone.
# * `--abort-on-collapse` with COLLAPSE_PATIENCE_EPOCHS=100. Measured against every arm that
#   escaped, the latest escape onset is epoch 40 (`bxprec`), so a patience of 100 is 2.5x the
#   observed worst case and would have stopped `ibprec` at epoch 100 while sparing every arm
#   that recovered. It is a budget guard, not a fix.
# * `ibprec` additionally takes `--ib-anchors three`, the secondary anchor set from the
#   original design: alpha-isopropylmalate plus 2-keto-isovalerate plus L-valine, giving six
#   concentration rows instead of two.
#
# READ THIS AS A REPAIR ATTEMPT, NOT A CLEAN CONTRAST. `ibprec` moves two flags at once, so a
# 9b `ibprec` that escapes does not say which of them did it; separating them needs the two
# single-flag cells. The pooled and scalar arms move one flag each and stay comparable.
#
# NEW INSTRUMENTATION, so the next failure is readable from the result file alone: every
# result now carries `target_standardization` (the train-split mean and sd actually used), and
# every precursor-concentration arm carries `precursor_concentration` (the across-genotype sd
# of the whole-cell log10 concentration at each anchor, plus the fraction of predictions
# pinned at the window's ceiling and floor). A flat pool and a dead affine are different
# numbers there, and wave 9 could not tell them apart without the checkpoint.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave9b_ibprec_e200_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w9b-ibprec-s2700
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
export COLLAPSE_PATIENCE_EPOCHS=100
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave7
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE7_EPOCHS:-200}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection_anchored --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.1 \
    --seed 2700 --epochs "$EPOCHS" --crps --graph-reg --abort-on-collapse \
    --head isobutanol --ib-readout flux_precursor_concentration --ib-anchors three \
    --conc-space mixture --conc-upper-m 1.0 \
    --wandb-group gh_wave9b_ib --tag _w9bibprec \
    --out wave9b_ibprec_e200_s2700.json
