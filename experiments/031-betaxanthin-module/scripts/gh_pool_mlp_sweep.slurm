#!/bin/bash
# experiments/031-betaxanthin-module/scripts/gh_pool_mlp_sweep.slurm
# [[experiments.031-betaxanthin-module.scripts.target_go_models]]
#
# The pool through a perceptron on CPU: target_go_models.py with the measured amino-acid
# pool as the ``store`` block, the ``store_mlp`` and ``go_store_mlp`` arms, one hidden
# width per pass (16, 64, 256, 1024), six seeds each, one result file per width
# (results/go_models_betaxanthin_pool_h<H>.json). The CPU sibling of rounds 180, 185 and
# 187, in minutes rather than GPU hours.
#
#   sbatch experiments/031-betaxanthin-module/scripts/gh_pool_mlp_sweep.slurm
#
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=031-poolmlp
#SBATCH --time=06:00:00
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
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
for H in 16 64 256 1024; do
  echo "== hidden $H"
  python experiments/031-betaxanthin-module/scripts/target_go_models.py \
    --target betaxanthin --store YPD_bx_aapool --arms store_mlp go_store_mlp \
    --mlp-hidden "$H" --workers 1 --threads 8 \
    --out "$EXPERIMENT_ROOT/031-betaxanthin-module/results/go_models_betaxanthin_pool_h$H.json" || exit 1
done
