#!/bin/bash
# experiments/031-betaxanthin-module/scripts/gh_target_forest_stack_store.slurm
# [[experiments.031-betaxanthin-module.scripts.target_forest_stack_store]]
#
# The message-passing readouts (ego nets on the gene graphs, GO as an incidence graph)
# on CPU: two-layer graph convolution over at most 9,000 nodes is seconds an epoch, so
# no card is taken. Every CPU diagnostic on the metabolic-module studies runs under
# slurm with a memory and CPU cap (2026-09-28): 8 cores and 32 GB here.
#
#   TARGET=betaxanthin BASE=YPD_bx_gonbr OUT=YPD_bx_gonbrforest sbatch experiments/031-betaxanthin-module/scripts/gh_target_forest_stack_store.slurm
#
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=031-stack
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/031-betaxanthin-module/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/031-betaxanthin-module/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
echo "host $(hostname); target ${TARGET:-betaxanthin}; epochs ${EPOCHS:-200}; DATA_ROOT=$DATA_ROOT"
python experiments/031-betaxanthin-module/scripts/target_forest_stack_store.py \
    --target "${TARGET:-betaxanthin}" --base "${BASE:-YPD_bx_gonbr}" --out "${OUT:-YPD_bx_gonbrforest}" --threads 8
