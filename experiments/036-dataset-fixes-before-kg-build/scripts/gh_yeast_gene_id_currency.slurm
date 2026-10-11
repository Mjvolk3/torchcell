#!/bin/bash
#SBATCH --job-name=036-geneid-measure
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/fix/yeast-gene-id-currency/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/fix/yeast-gene-id-currency/experiments/036-dataset-fixes-before-kg-build/slurm/output/%j_%x.out
#SBATCH --no-requeue
# experiments/036-dataset-fixes-before-kg-build/scripts/gh_yeast_gene_id_currency.slurm
# [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_yeast_gene_id_currency]]
#
# Issues #893/#930/#979 and #897-#900: classify every released Costanzo 2016 / Kuzmin
# 2020 ORF R64-4-1 no longer carries and count what each store's currency check does
# (yeast_gene_id_currency.py). Reads the 2 GB Data File S1, so it runs under slurm.
#
#   sbatch experiments/036-dataset-fixes-before-kg-build/scripts/gh_yeast_gene_id_currency.slurm
echo "Running on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
PROJECT_ROOT="/home/michaelvolk/Documents/projects/torchcell.worktrees/fix/yeast-gene-id-currency"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"
set -a; source "$PROJECT_ROOT/.env"; set +a
export WANDB_MODE=disabled
python experiments/036-dataset-fixes-before-kg-build/scripts/yeast_gene_id_currency.py || exit 1
echo "finished at $(date)"
