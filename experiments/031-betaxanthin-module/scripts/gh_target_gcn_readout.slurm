#!/bin/bash
# experiments/031-betaxanthin-module/scripts/gh_target_gcn_readout.slurm
# [[experiments.031-betaxanthin-module.scripts.target_gcn_readout]]
#
# The message-passing readouts (ego nets on the gene graphs, GO as an incidence graph)
# on CPU: two-layer graph convolution over at most 9,000 nodes is seconds an epoch, so
# no card is taken. Every CPU diagnostic on the metabolic-module studies runs under
# slurm with a memory and CPU cap (2026-09-28): 8 cores and 32 GB here.
#
#   TARGET=betaxanthin sbatch experiments/031-betaxanthin-module/scripts/gh_target_gcn_readout.slurm
#   EPOCHS=600 TARGET=betaxanthin sbatch ...   (the first run, job 2955, selected epoch 199
#                                              of 200 on every GCN seed: not converged)
#
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=031-gcn
#SBATCH --time=02:00:00
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
python experiments/031-betaxanthin-module/scripts/target_gcn_readout.py \
    --target "${TARGET:-betaxanthin}" --threads 8 --epochs "${EPOCHS:-200}" \
    ${GCN_OUT_SUFFIX:+--out-suffix "$GCN_OUT_SUFFIX"}
