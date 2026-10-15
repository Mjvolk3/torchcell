#!/bin/bash
# experiments/030-solid-growth-multi/scripts/gh_warm_fraction_caches_030.slurm
# [[experiments.030-solid-growth-multi.scripts.gh_warm_fraction_caches_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/gh_warm_fraction_caches_030.slurm
#
# Write the data module's split cache for the scaling sweep's subset pools (1/4, 1/16,
# 1/64 of S3Q by query pair; subset_s3q_fractions_030.py), seeds 0 to 2, INTO THE S3Q POOL
# STORE's data_module_cache, by building the dataset over the store (dataset.root_rel is
# the store's absolute path, which the trainer joins with DATA_ROOT as an absolute path).
# The store already holds every record of every subset, so one store serves the whole
# sweep; what each pool needs is its own cache file (the name carries the pool hash), and
# a 1-GPU Delta job would compute it alone but a 4-GPU one would race. CPU only.
#
#   sbatch experiments/030-solid-growth-multi/scripts/gh_warm_fraction_caches_030.slurm
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=030-warm-frac
#SBATCH --time=06:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
EXP="$PROJECT_ROOT/experiments/030-solid-growth-multi"
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME host=$(hostname) started=$(date)"
echo "project=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; source "$PROJECT_ROOT/.env"; set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export PYTHONUNBUFFERED=1
STORE="$DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/001-multi-build-s3q"
[[ -f "$STORE/processed/lmdb/STORE.json" ]] || { echo "ERROR: no pool store at $STORE" >&2; exit 1; }
cd "$PROJECT_ROOT/experiments" || exit 1
STATUS=0
for f in 4 16 64; do
  echo "================ 1/$f  $(date) ================"
  python "$EXP/scripts/warm_split_cache_030.py" --config-name cgt_030_s3_q_tok_embfit_004 \
    "subset.indices=subset_S3Q_frac${f}_indices_030.json.gz" "dataset.root_rel=$STORE" \
    '+warm.seeds=[0,1,2]' || { echo "WARM 1/$f FAILED"; STATUS=2; }
done
ls -la "$STORE/data_module_cache"
echo "finished=$(date) exit=$STATUS"
exit $STATUS
