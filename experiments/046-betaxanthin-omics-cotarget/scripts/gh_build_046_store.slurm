#!/bin/bash
# experiments/046-betaxanthin-omics-cotarget/scripts/gh_build_046_store.slurm
# [[experiments.046-betaxanthin-omics-cotarget.scripts.gh_build_046_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/046-betaxanthin-omics-cotarget/scripts/gh_build_046_store.slurm
#
# Build the experiment 046 store (Cachera betaxanthin, Mulleder amino acids, Kemmeren
# expression, Messner proteome of KG 4.0; one sample per served experiment) from the served
# graph on GilaHyper, under slurm (DB work never runs as a login-shell process). Submit from
# the repository root of the 046 worktree:
#
#   sbatch experiments/046-betaxanthin-omics-cotarget/scripts/gh_build_046_store.slurm
#
# Sizing: 15,620 served experiments against 043's 173,617 (job 5856: 153 GB raw LMDB,
# 2.4 h after the raw stage, 256 GB peak RSS through the aggregation copy). The Kemmeren
# records carry 6,169 values and Messner's 1,838, so the raw store here should be a few GB;
# 64 GB and four hours are generous.
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=046-build-store
#SBATCH --time=04:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/046-betaxanthin-omics-cotarget/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/046-betaxanthin-omics-cotarget/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
python experiments/046-betaxanthin-omics-cotarget/scripts/build_046_store.py
echo "done $(date)"
