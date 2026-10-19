#!/bin/bash
#SBATCH --job-name=044-isolate-protid
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=experiments/044-organism-similarity/slurm/output/%x_%j.out
#SBATCH --error=experiments/044-organism-similarity/slurm/output/%x_%j.out
# experiments/044-organism-similarity/scripts/gh_06_isolate_protein_identity.slurm
# [[experiments.044-organism-similarity.scripts.gh_06_isolate_protein_identity]]
#
# Per-isolate protein identity vs S288C over the reference ORFs (step "isolates" of
# 06_isolate_protein_identity.py). STEPS overrides the step list, e.g. STEPS=scog after
# the eggNOG-mapper run. Submit from the worktree root:
#   sbatch experiments/044-organism-similarity/scripts/gh_06_isolate_protein_identity.slurm
echo "Running on $(hostname) at $(date)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell
ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"
set -a; source "$ROOT/.env"; set +a
python experiments/044-organism-similarity/scripts/06_isolate_protein_identity.py ${STEPS:-isolates} || exit 1
echo "finished at $(date)"
