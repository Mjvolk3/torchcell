#!/bin/bash
#SBATCH --job-name=044-mmseqs-rbh
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=experiments/044-organism-similarity/slurm/output/%x_%j.out
#SBATCH --error=experiments/044-organism-similarity/slurm/output/%x_%j.out
# experiments/044-organism-similarity/scripts/gh_03_mmseqs_rbh.slurm
# [[experiments.044-organism-similarity.scripts.gh_03_mmseqs_rbh]]
#
# All-vs-all mmseqs2 over the per-gene proteomes, then reciprocal best hits and
# results/pairwise_similarity.csv. Submit from the worktree root:
#   sbatch experiments/044-organism-similarity/scripts/gh_03_mmseqs_rbh.slurm
echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell
ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"
set -a; source "$ROOT/.env"; set +a
python experiments/044-organism-similarity/scripts/03_mmseqs_rbh.py search || exit 1
python experiments/044-organism-similarity/scripts/03_mmseqs_rbh.py rbh || exit 1
echo "finished at $(date)"
