#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave9_aa_conc_sm_e200_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.wave9-isobutanol]]
#
# WAVE 9 BLOCK 4: AMINO-ACID concentration on SM bounds, seed 2700.
#
# THE NULL ARM OF THE PAIR. Mulleder's strains are prototrophs on synthetic minimal agar, so
# the medium the shipped bounds approximate is the right one for this target and applying SM
# explicitly should leave the arm where it is. It is not a pure no-op, though: SM's uptake
# policy takes wild-type growth from 0.0858 to 0.3138, so this cell also says how much of the
# arm's score is sensitive to the absolute uptake magnitudes at all.
#
# EVERY OTHER FLAG MATCHES gh_wave7_aa_conc_e200_s2700.slurm, so the contrast is the medium
# alone. If this arm MOVES by more than its 0.0094 paired seed sd, the wiring changed
# something it should not have and the cell is invalid rather than informative.
#
# WHAT THIS CHANGES. Every 026 and 027 arm to date ran on the SHIPPED yeast-GEM 9.0.2
# boundary, where all 20 amino-acid exchanges have lower_bound 0, i.e. effectively synthetic
# minimal, with a wild-type growth rate of 0.0858 1/h. `--apply-media dataset` closes every
# exchange and opens exactly the medium the dataset was grown in, through
# `media_to_bounds(FBA_MEDIA[name], model).apply(model)`, BEFORE the stoichiometric tensors
# are taken, and records the wild-type growth rate before and after in flux_meta.media.
#
# MEASURED ON THE GEM, 2026-09-14, all components resolving: YPD-approx opens 45 exchanges
# and takes wild-type growth 0.0858 -> 0.5352; SM opens 25 and takes it 0.0858 -> 0.3138; SC
# opens 47 and takes it 0.0858 -> 0.5428. So SM is NOT a no-op against the shipped bounds --
# the shipped boundary is tighter than the recipe's uptake policy -- and the amino-acid cell
# below is therefore a real arm, not only a wiring check.
#
# PRE-REGISTERED READ: the prediction is ASYMMETRIC and that is the point. The amino acids are
# grown on synthetic minimal, so their arm should move little; betaxanthin is grown on YPD,
# which supplies the tyrosine its cassette draws on, so its arm should move by more than its
# 0.026 paired seed sd in either direction. SUCCESS/KILL: if betaxanthin does not move, the
# wrong-medium explanation for the betaxanthin null is falsified and has to be retracted from
# the memory that states it.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave9_aa_conc_sm_e200_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w9-aa-conc-sm-s2700
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
    --arm flux_projection_anchored --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.1 \
    --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --head mulleder19 --aa-readout flux_concentration --conc-space mixture --conc-upper-m 1.0 \
    --apply-media dataset \
    --wandb-group gh_wave9_media --tag _w9aaconc_sm \
    --out wave9_aa_conc_sm_e200_s2700.json
