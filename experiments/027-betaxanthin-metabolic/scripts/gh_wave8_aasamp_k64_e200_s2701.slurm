#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave8_aasamp_k64_e200_s2701.slurm
# [[experiments.027-betaxanthin-metabolic.wave8-sampling]]
#
# WAVE 8 STEP 3 (2026-09-13): THE K-DRAW CONTRAST ON THE AMINO-ACID CONCENTRATION READOUT,
# K = 64, seed 2701.
#
# THE QUESTION. "we will want to also know if sampling helps for amino acid prediction from
# amino acid dataset." The betaxanthin K-draw contrast closed NULL at 200 epochs, n = 3
# (K = 128 minus K = 1, paired by seed: +0.060, -0.009, -0.036). The amino-acid target is
# where the latent should matter, but the wave-8 concentration readout could not answer it:
# `thermo_log_c` was a function of the genotype context alone, so K draws gave K copies of
# one concentration (Finding 5 in the note above).
#
# WHAT IS NEW, AND IT IS ONE TERM. `--flux-log-c-from-latent` makes the anchored
# concentration read the drawn latent as well as the context,
#     log c = lo + (hi - lo) sigmoid(base + W context + U z),
# with U zero-initialized, so the flag-on cell starts bit-identical to the flag-off one and
# the latent path is learned rather than injected as noise. The reaction energies, the
# second-law hinge and the dissipation are then computed per draw, from the same z the flux
# is built from, so the hinge prices sign consistency between the drawn v and the drawn
# log c. The readout writes one [19, 2] row per draw; FluxDrawEnsemble reduces them by the
# same moment match the betaxanthin cells use, and the training loss is the mean over draws
# of the per-draw CRPS against the target repeated K times.
#
# THE DESIGN, THREE COLUMNS PAIRED BY SEED (2700, 2701, 2702).
#   concver  : wave8_aa_concver_e200_s<seed>.json, flag OFF, K = 1 (already run:
#              0.0580 / 0.0725 / 0.0554 mean test Spearman at s2700 / 2701 / 2702).
#   aasamp k1: this recipe with --flux-latent-stochastic --flux-log-c-from-latent
#              --flux-draws 1. Against concver it measures what the stochastic,
#              draw-dependent concentration costs on its own.
#   aasamp k64: the same with --flux-draws 64. THE CONTRAST is k64 minus k1, both flag on,
#              paired by seed; only the draw count differs.
# K = 64 rather than the betaxanthin cells' 128: the aa head is 19-wide, the 200-epoch aa
# cell already takes 3.3 h, and the betaxanthin K = 128 cost 1.22 to 1.30x, so 64 is
# expected (not measured) to land under 1.3x.
#
# EVERYTHING ELSE MATCHES gh_wave8_aa_concver_e200_s2701.slurm: the verified reaction-energy
# table, constraint scale 0.1 over a 20-epoch warmup, the widened 1 M ceiling, the mixture
# concentration space, 200 epochs, the same seed, and the same scoring rule (argmax of the
# centered 5-epoch moving average of validation Pearson).
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave8_aasamp_k64_e200_s2701.slurm
#
#SBATCH -p main
#SBATCH --mem=100g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w8-aasamp-k64-s2701
#SBATCH --time=12:00:00
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
    --arm flux_projection_anchored --kcat-tier predicted --thermo-source verified \
    --constraint-warmup-epochs 20 --constraint-scale 0.1 \
    --seed 2701 --epochs "$EPOCHS" --crps --graph-reg \
    --head mulleder19 --aa-readout flux_concentration --conc-space mixture --conc-upper-m 1.0 \
    --flux-latent-stochastic --flux-log-c-from-latent --flux-draws 64 \
    --wandb-group gh_wave8_aasamp --tag _w8aasamp \
    --out wave8_aasamp_k64_e200_s2701.json
