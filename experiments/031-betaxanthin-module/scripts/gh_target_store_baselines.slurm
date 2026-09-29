#!/bin/bash
# experiments/031-betaxanthin-module/scripts/gh_target_store_baselines.slurm
# [[experiments.031-betaxanthin-module.scripts.target_store_baselines]]
#
# The CPU ladder (ridge, cosine nearest neighbors, random forest, and opt-in gradient
# boosting) on a subset of representations, under slurm with a CPU and memory cap
# (2026-09-28: every diagnostic on the metabolic-module studies is an sbatch job).
#
#   TARGET=betaxanthin ONLY="aa_pool go go_nbr" MODELS="forest xgb" \
#     OUT=experiments/031-betaxanthin-module/results/store_baselines_betaxanthin_xgb.json \
#     sbatch experiments/031-betaxanthin-module/scripts/gh_target_store_baselines.slurm
#
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=031-ladder
#SBATCH --time=04:00:00
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
: "${TARGET:=betaxanthin}"
: "${OUT:?set OUT to the results json path}"
echo "host $(hostname); target $TARGET; only ${ONLY:-all}; models ${MODELS:-ridge knn forest}; out $OUT"
python experiments/031-betaxanthin-module/scripts/target_store_baselines.py \
    --target "$TARGET" --threads 8 --out "$OUT" \
    ${ONLY:+--only $ONLY} ${MODELS:+--models $MODELS}
