#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_cpu_score_fcl_027.slurm
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
#
# E4 of the metabolic review: the 027 wave-7 dumps read through Merzbacher's own three-class
# protocol, so the comparison lands on their terms as well as ours.
#
# CONTROL, run first and not optional: the 020 dumps are re-scored through the same adapter
# and every number in `020-cachera-betaxanthin/results/merzbacher_head_to_head.json` must
# come back exactly. The script raises otherwise and reports nothing.
#
# PRE-REGISTERED READ. MCC, accuracy, high-producer recall, fraction called medium and
# top-50 precision per arm, mean +/- SEM over seeds, under the train-pool scale and the
# rank-matched view, against FCL's 0.232 / 0.700 / 0.18 / 0.948.
#   SUCCESS: an arm's high-producer-recall CI clears 0.295, Merzbacher's ceiling over all
#            sixteen of their model x rebalancing cells.
#   KILL:    the fraction called medium is ~0.95 for every arm. We reproduced their failure
#            mode, and the write-up says so in those words.
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=027-fcl-score
#SBATCH --time=01:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
echo "host $(hostname); DATA_ROOT=$DATA_ROOT"

python experiments/027-betaxanthin-metabolic/scripts/score_fcl_027.py \
    --dumps "$DATA_ROOT/test-predictions-027-wave7" --dump-format 027
