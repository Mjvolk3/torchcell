#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave7_fill_proj_boxend_cstr_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.projection-head]]
#
# WAVE 7 fill (2026-09-12): flux_projection_anchored_k20_boxend, constraint scale 0.1 (everything on), seed 2700, union_external deltaG.
#
# Anchored thermodynamics on the union deltaG table, the enzyme-capacity bound through the
# predicted per-gene abundance head, the forward protein-pool budget and the Gibbs
# dissipation limit, on top of a flux that is balanced and in-box by construction.
#
# THE BOX END WITHOUT THE SIGN TIGHTENING: the control that says which constraint cost the
# signal. Measured at initialization on yeast-GEM with the REAL gene universe, untrained,
# eight samples:
#
#   head                       frac |v| < 1e-6   |v| mean   balance   box    second law
#   projection (balance end)   0.119             1.159      5.4e-9    0.527  0.508
#   projection_k20_box         0.441             0.355      2.7e-1    0.000  0.332
#   projection_k20_box_tdg     0.680             0.067      2.6e-1    0.000  0.000
#
# Ending on the clip, with NO thermodynamics in the box at all, already takes the flux from
# 11.9 percent zeros to 44.1 percent and the balance residual from 5e-9 to 0.27. The sign
# tightening then adds the rest. So the box end alone may account for most of the signal
# loss, and this arm is what separates the two.
#
# If this arm also scores at chance, the cause is respecting the bounds, and the balance-end
# head's performance came from flux the model's own box forbids. If it keeps the signal, the
# sign tightening is the cause. Either answer is worth a cell.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave7_fill_proj_boxend_cstr_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w7-fill-proj-boxend-cstr-s2700
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
    --arm flux_projection_anchored_k20_boxend --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.1 --head-normalize \
    --wandb-group gh_wave6_proj --tag _w7fill \
    --out wave6_proj_flux_projection_anchored_k20_boxend_ue_s2700.json
