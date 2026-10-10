#!/bin/bash
# experiments/030-solid-growth-multi/scripts/gh_make_pool_store_030.slurm
# [[experiments.030-solid-growth-multi.scripts.gh_make_pool_store_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/gh_make_pool_store_030.slurm
#
# Build the S3Q pool store of the 030 build on GilaHyper (make_pool_store_030.py): 1.12M
# random reads of the local-NVMe LMDB at about 1 ms each plus zlib-6 (0.5 ms per record),
# so under an hour on one core; the LMDB write is sequential. CPU only. 16 GB holds the
# two LMDB maps (the source is memory-mapped read-only, pages come and go) and the pool
# index. --mem is sized from the CPUs (MaxMemPerCPU 4.1 GB; gh_smoke_dataset_token.slurm).
#
#   mkdir -p experiments/030-solid-growth-multi/slurm/output
#   sbatch experiments/030-solid-growth-multi/scripts/gh_make_pool_store_030.slurm
#   INDICES=subset_S3Q_indices_030.json.gz SUFFIX=s3q are the defaults; override with --export.
#SBATCH -p main
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=030-poolstore-s3q
#SBATCH --time=06:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
INDICES="${INDICES:-subset_S3Q_indices_030.json.gz}"
SUFFIX="${SUFFIX:-s3q}"
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME host=$(hostname) started=$(date)"
echo "project=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null)) indices=$INDICES suffix=$SUFFIX"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; source "$PROJECT_ROOT/.env"; set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT" || exit 1
python experiments/030-solid-growth-multi/scripts/make_pool_store_030.py \
  --indices "$INDICES" --suffix "$SUFFIX"
STATUS=$?
DEST="$DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/001-multi-build-$SUFFIX"
[[ -d "$DEST" ]] && { du -sh "$DEST"/processed "$DEST"/data_module_cache; ls -la "$DEST/processed/lmdb"; }
echo "finished=$(date) exit=$STATUS"
exit $STATUS
