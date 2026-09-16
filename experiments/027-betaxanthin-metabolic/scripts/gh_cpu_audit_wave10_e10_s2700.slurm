#!/bin/bash
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=027-audit-e10-s2700
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
python experiments/027-betaxanthin-metabolic/scripts/audit_top_strain_feasibility.py \
    --result wave10_warmstart_frozen_e10_k100_boxforce_e10_s2700.json \
    --out audit_wave10_frozen_e10_s2700.json
