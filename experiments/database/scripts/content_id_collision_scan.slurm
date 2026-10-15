#!/bin/bash
#SBATCH --job-name=content-id-scan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# experiments/database/scripts/content_id_collision_scan.slurm
# [[experiments.database.scripts.content_id_collision_scan]]
#
# One store per job for the three dev stores lane Q's scan skipped as too large
# (dmf_costanzo2016 and dmi_costanzo2016 at 77.8 GB, env_chemgen_hoepfner2014 at
# 125.7 GB). They are the three biggest datasets in the graph, so whether their records
# collide on the Experiment content id is the open half of lane Q's result.
#
# The store is read READ-ONLY (lmdb readonly, lock=False); nothing under $DEV_DATA_ROOT
# is written. The result goes to its own JSON per store so two jobs never write one file,
# and the scan is resumable because the scanner keeps results already in --out.
#
#   sbatch --export=ALL,TORCHCELL_SRC=<wt>,SLUG=env_chemgen_hoepfner2014 \
#       experiments/database/scripts/content_id_collision_scan.slurm
#
# Variables: TORCHCELL_SRC (the checkout to run, PYTHONPATH), SLUG (the store),
# DEV_DATA_ROOT (/scratch/projects/torchcell-scratch), OUT (the result JSON).
# One worker: the pool is per store, so a single store is scanned by one process and the
# 16 CPUs are for the page-cache reads and the unpickling, not for parallelism.
# `/usr/bin/time -v` prints the max RSS, which is what decides whether 64 G is enough.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -euo pipefail

TORCHCELL_SRC="${TORCHCELL_SRC:?set TORCHCELL_SRC=<checkout to run>}"
SLUG="${SLUG:?set SLUG=<store slug>}"
DEV_DATA_ROOT="${DEV_DATA_ROOT:-/scratch/projects/torchcell-scratch}"
OUT="${OUT:-$TORCHCELL_SRC/experiments/database/results/content_id_collision_scan_$SLUG.json}"
JOB="${SLURM_JOB_ID:-nojob}"

cd "$TORCHCELL_SRC"
export PYTHONPATH="$TORCHCELL_SRC"

echo "Running on $(hostname) at $(date); job $JOB"
echo "store $SLUG under $DEV_DATA_ROOT; out $OUT"
ls -la "$DEV_DATA_ROOT/data/torchcell/$SLUG/processed/lmdb/data.mdb"

/usr/bin/time -v python experiments/database/scripts/content_id_collision_scan.py \
    --data-root "$DEV_DATA_ROOT" --out "$OUT" --workers 1 --only "$SLUG"

# The scanner prints its own per-store summary line (records, collision groups, records
# lost in the graph, wall seconds) to stderr, which lands in this .out above.
echo "scan finished at $(date)"
