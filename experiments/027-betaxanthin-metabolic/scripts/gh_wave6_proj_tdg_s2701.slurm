#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_tdg_s2701.slurm
# [[experiments.027-betaxanthin-metabolic.projection-head]]
#
# WAVE 6 (2026-09-11): flux_projection_anchored_tdg, constraint scale 0.1, seed 2701, union_external deltaG.
#
# Anchored thermodynamics on the union deltaG table, the enzyme-capacity bound through the
# predicted per-gene abundance head, the forward protein-pool budget and the Gibbs
# dissipation limit, on top of a flux that is balanced and in-box by construction.
#
# THE SECOND LAW AS A BOX. Jobs 1689, 1690 and 1708 made mass balance exact by
# construction and left the thermodynamics soft, and the soft term closed nothing: on job
# 1708's top-100 predicted strains, at a constraint scale of 0.1, every strain met the
# balance criterion at every tolerance while thermo_model_violation_frac still read a median
# of 0.405. The Genesis criterion therefore failed on the deltaG half in every arm run so
# far. A weight traded against the data term cannot deliver "95 percent of the top 100
# violate nothing"; a parameterization can.
#
# Here the sign of each reaction's deltaG becomes a BOUND on its flux (dG > +eps clamps ub
# to 0, dG < -eps clamps lb to 0, near-equilibrium reactions are left alone), the box is
# per sample because deltaG carries the learned concentration term, and twenty alternations
# end on the clip. Bounds, enzyme capacity and direction are then all exact and the balance
# residual is the only reported leftover. Measured on yeast-GEM at initialization, untrained:
# second-law violation fraction 0, box violation fraction 0, balance ratio median under 1e-2
# on every sample (tests/torchcell/metabolism/test_flux_layer.py).
#
# Read per cell: test Spearman on the pinned genes against pooled at the same seed, the
# escape epoch and whether it sustained, and the top-100 fractions for balance <= 1e-2,
# box == 0, thermo == 0, and BOTH, which is the Genesis number.
#
# NOTE on c_thermo in the log: its epsilon makes a flux pinned to zero score 1, so this arm
# runs a HIGH c_thermo while violating nothing. The feasibility block is the readout.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_tdg_s2701.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w6-proj-tdg-s2701
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
    --arm flux_projection_anchored_tdg --seed 2701 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.1 --head-normalize \
    --wandb-group gh_wave6_proj --tag _w6proj \
    --out wave6_proj_flux_projection_anchored_tdg_ue_s2701.json
