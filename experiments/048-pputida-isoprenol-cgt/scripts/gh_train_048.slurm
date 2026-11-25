#!/bin/bash
# experiments/048-pputida-isoprenol-cgt/scripts/gh_train_048.slurm
# [[experiments.048-pputida-isoprenol-cgt.scripts.gh_train_048]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/gh_train_048.slurm
#
# One arm group of experiment 048 per job, so the groups queue independently
# instead of one job holding a card for every arm in series. ARM selects the
# group and run_mvp_arms.sh holds what each one is. Submit from the repo root:
#
#   sbatch --export=ALL,ARM=design experiments/048-pputida-isoprenol-cgt/scripts/gh_train_048.slurm
#   sbatch --export=ALL,ARM=cycle  experiments/048-pputida-isoprenol-cgt/scripts/gh_train_048.slurm
#
# ARM is one of: design, design_flux, cycle, cycle_flux,
# design_fidelity, cycle_fidelity, design_random, all.
# SEEDS, EPOCHS, PATIENCE and EMB override the defaults.
#
# One GPU and 8 cores, deliberately not a half node: the campaign is 465 records
# and the model is small, so the job is latency-bound on the dataset load rather
# than throughput-bound, and the 031 program is the queue's priority.
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=048-train
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/048-pputida-isoprenol-cgt/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/048-pputida-isoprenol-cgt/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export EXPERIMENT_ROOT="$PWD/experiments"
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-torchcell}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=0
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  arm ${ARM:-all}  $(date)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true
ARM="${ARM:-all}" bash experiments/048-pputida-isoprenol-cgt/scripts/run_mvp_arms.sh
echo "done $(date)"
