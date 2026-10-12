#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave5_g1_flux_capacity_w20_s0p1.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# WAVE 5 (2026-09-10), RESTART on corrected units. The shipped deltaG CSVs are kcal/mol
# mislabeled kJ/mol (yeast-GEM copied yETFL without converting; slope 4.187 at r 0.987 over
# 1,927 metabolites against the exact 4.184 kJ/kcal), so the union table this cell reads was
# adding kcal-scale shipped values to kJ-scale eQuilibrator values. constraints.py now
# converts at parse time. The pre-fix run of this cell was cancelled on the plateau.
#
# WAVE 5 (2026-09-10), one GilaHyper GPU, seed 2700, 80 epochs, the recipe the pooled arm
# escapes the mean plateau under (CRPS head + graph regularizer 6.5e-5, betaxanthin only,
# two layers), no collapse abort.
#
# G1. The abundance head with the predicted kcat tier (148 measured / 3,410 predicted / 170
# organism-default catalytic units, against 148 / 0 / 3,580 before) and the union deltaG
# table (3,722 of 4,131 reactions priced, against 3,505), warmed up over 20 epochs to a
# tenth of the configured constraint weight.
#
# MEASURED CONTEXT, not a claim about this run: under the same recipe at 40 epochs the two
# flux arms sat at a validation prediction spread of 0.000 with a validation DATA loss of 1
# to 5 (jobs 1636, 1637), while the pooled arm reached spread 0.33 and val Pearson +0.24 at
# a data loss of 0.49 (job 1634).
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave5_g1_flux_capacity_w20_s0p1.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w5-flux_capacity-warm0.1-kj
#SBATCH --time=04:00:00
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
export NUM_WORKERS=3
# The abort is off in this wave anyway (--abort-on-collapse is not passed); the patience is
# pinned so a stray environment cannot reintroduce the ten-epoch abort that killed every run
# which would later have escaped.
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave5
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE5_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_capacity --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union --constraint-warmup-epochs 20 --constraint-scale 0.1 \
    --wandb-group gh_wave5_000 --tag _w5 \
    --out wave5_g1_flux_capacity_w20_s0p1_s2700_kj.json
