#!/bin/bash
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=027-probe-nocstr
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
python experiments/027-betaxanthin-metabolic/scripts/probe_flux_genotype_spread.py \
    --result wave5_g7_flux_capacity_nocstr_s2700.json \
    --out probe_flux_spread_nocstr_s2700.json
