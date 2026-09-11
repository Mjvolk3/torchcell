#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave3_flux_capacity.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# WAVE 3 (2026-09-09), arm "flux_capacity", seed 2700, one GilaHyper GPU, the 020 recipe (CRPS
# head, hard attention mask on all nine graphs, nine perturbation heads) with NO collapse
# abort and the longest budget the window allows. The August 020 grid left the mean plateau
# at epochs 117 to 157 on this recipe; the question is whether HEAD does the same, and for
# flux_capacity what the abundance head's capacity and budget diagnostics do once it does.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave3_flux_capacity.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w3-flux_capacity
#SBATCH --time=01:45:00
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
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave3
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE3_EPOCHS:-150}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_capacity --seed 2700 --epochs "$EPOCHS" --crps --attention-mask --pert-heads 9 \
    --wandb-group gh_wave3_000 --tag _w3 \
    --out wave3_flux_capacity_s2700.json
