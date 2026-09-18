#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_graph_baselines.slurm
# [[experiments.028-gene-essentiality.scripts.graph_baselines]]
#
# One arm set of graph_baselines.py on one GPU. The base, extra and split arm sets run in
# an hour on CPU; the relational set (RGCN and RGAT over nine typed edge sets, GIN,
# GraphSAGE) does not print its first arm within an hour at ten cores (2026-09-17), so it
# takes a card.
#
#   ARMS=relational sbatch experiments/028-gene-essentiality/scripts/gh_graph_baselines.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=028-graph
#SBATCH --time=06:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
echo "host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}; arms ${ARMS:?set ARMS}"
python experiments/028-gene-essentiality/scripts/graph_baselines.py --arms "$ARMS"
