#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave8_samp_k16_s2703.slurm
# [[experiments.027-betaxanthin-metabolic.wave8-sampling]]
#
# WAVE 8 (2026-09-12), arm (a) of the K-DRAW CONTRAST, K = 16 at seed 2703.
#
# THE QUESTION. Flux Cone Learning samples each deletion's flux cone 124 times with
# OptGPSampler at a thinning of 5,000 and reads the phenotype off the cone's geometry; our
# layer draws ONE latent per genotype per forward pass. This pair asks whether more draws
# per pass raise the metabolic signal, on the one 027 readout the Step 0 diagnostic showed
# can respond to a draw at all.
#
# WHY BOTH ARMS SAMPLE. Every wave-6 and wave-7 projection cell ran `latent_stochastic`
# OFF, so its spread head never received a gradient and its sigma stayed at exp(-3) exactly
# (results/wave8_latent_spread.json). Pairing K = 16 against one of those cells would
# confound the draw count with whether the model samples at all, so BOTH arms of this
# contrast carry --flux-latent-stochastic and differ only in --flux-draws. The
# deterministic cells 1776 to 1778, same recipe and same seeds, are the third column: what
# turning the draw on costs on its own.
#
# THE RECIPE IS gh_wave7_fill_proj_projection_s2703.slurm VERBATIM, plus the two draw
# flags. Plain projection, constraint scale exactly 0.0, 80 epochs, predicted kcat, union
# thermodynamics, normalized head, CRPS and the graph regularizer.
#
# SCORED ON: pinned Merzbacher test Spearman at the validation-selected epoch (centered
# 5-epoch moving average), paired by seed against its K = 1 partner. Never a max over
# epochs.
#
#   cd <worktree-root> && bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave8_samp_k16_s2703.slurm
#
#SBATCH -p main
#SBATCH --mem=80g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w8-samp-k16-s2703
#SBATCH --time=16:00:00
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
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave8
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE8_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection --seed 2703 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --flux-latent-stochastic --flux-draws 16 \
    --wandb-group gh_wave8_samp --tag _w8samp \
    --out wave8_samp_k16_s2703.json
