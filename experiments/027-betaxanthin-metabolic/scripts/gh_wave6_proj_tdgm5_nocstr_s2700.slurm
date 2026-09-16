#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_tdgm5_nocstr_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.projection-head]]
#
# WAVE 6 (2026-09-11): flux_projection_anchored_tdgm5, constraint scale 0.0, seed 2700, union_external deltaG.
#
# Anchored thermodynamics on the union deltaG table, the enzyme-capacity bound through the
# predicted per-gene abundance head, the forward protein-pool budget and the Gibbs
# dissipation limit, on top of a flux that is balanced and in-box by construction.
#
# THE 5 kJ/mol MARGIN, the direct fix for what the diagnosis found. From job 1725's
# checkpoint, on the top-100 strains: 70.6 percent of reactions sit below 1e-6 in absolute
# flux against 11.9 percent for the untightened projection head, and the median absolute
# flux is 0.0 against 1.02. The alternating projection's residual falls smoothly to 2e-6
# rather than oscillating, so the feasible set is reachable and simply concentrated near
# the origin: the head is feasible and has almost nothing left to vary per genotype.
#
# The margin constrains only reactions at least 5 kJ/mol from equilibrium and leaves the
# rest bidirectional. Hypothesis (untested at submission): this recovers flux density and
# with it the phenotype signal, without giving up the exactness, because the reactions it
# releases are the ones whose direction the tabulated energy could not resolve anyway.
#
# Read against job 1725 (same arm family, same seed, same scale, no margin): does the
# Genesis fraction hold near 0.98 while test Spearman comes back toward pooled's 0.0717?
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_tdgm5_nocstr_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w6-proj-tdgm5-nocstr-s2700
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/027-betaxanthin-metabolic/wandb
export WANDB_PROJECT=torchcell_027_bxfx
export NUM_WORKERS=2
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave6
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE6_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection_anchored_tdgm5 --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave6_proj --tag _w6proj \
    --out wave6_proj_flux_projection_anchored_tdgm5_ue_nocstr_s2700.json
