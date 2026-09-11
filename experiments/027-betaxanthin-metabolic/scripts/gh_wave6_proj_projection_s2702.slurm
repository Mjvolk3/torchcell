#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_projection_s2702.slurm
# [[experiments.027-betaxanthin-metabolic.projection-head]]
#
# WAVE 6 (2026-09-10), cell P1: the PROJECTION head at a constraint scale of EXACTLY ZERO.
#
# THE PARITY READ. Wave 5 established that the flux path can escape the betaxanthin mean
# plateau once the head is normalized and the constraints are off: job 1673 escaped at epoch
# 33 and scored test Spearman 0.0747 at seed 2700, against the pooled control's 0.0717 (job
# 1674) at the same seed. The projection head changes the parameterization, not the physics
# weight, so this cell asks whether making mass balance and the box exact by construction
# costs anything relative to the box head at the same zero weight. A projection that cannot
# reach parity at scale 0 cannot be read at scale 0.1 either.
#
# Compare to, at seed 2701: pooled 0.1058 (job 1680), box head no-constraint 0.1185 (job
# 1679). The 019 replicate study puts the between-seed spread near 0.04 in Pearson, so a gap
# under that is within noise at one seed, which is why this seed is run as well as 2700.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_projection_s2702.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w6-proj-nocstr-s2702
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
export NUM_WORKERS=2
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave6
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE6_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection --seed 2702 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave6_proj --tag _w6proj \
    --out wave6_proj_flux_projection_s2702.json
