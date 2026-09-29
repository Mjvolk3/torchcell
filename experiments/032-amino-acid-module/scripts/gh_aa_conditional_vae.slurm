#!/bin/bash
# experiments/032-amino-acid-module/scripts/gh_aa_conditional_vae.slurm
# [[experiments.032-amino-acid-module.scripts.aa_conditional_vae]]
#
# The conditional VAE on the amino-acid profile (posterior route against prior route),
# CPU under slurm with a cap: 8 cores, 32 GB.
#
#   sbatch experiments/032-amino-acid-module/scripts/gh_aa_conditional_vae.slurm
#
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=032-cvae
#SBATCH --time=03:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/032-amino-acid-module/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/032-amino-acid-module/slurm/output/%x_%j.out

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
echo "host $(hostname); DATA_ROOT=$DATA_ROOT"
python experiments/032-amino-acid-module/scripts/aa_conditional_vae.py --threads 8 --epochs "${EPOCHS:-300}"
