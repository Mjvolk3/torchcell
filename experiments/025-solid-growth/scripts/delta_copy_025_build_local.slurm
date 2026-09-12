#!/bin/bash
# experiments/025-solid-growth/scripts/delta_copy_025_build_local.slurm
# [[experiments.025-solid-growth.scripts.sync_taiga_025_build]]
#
# Copy the 025 full build from the Taiga NFS mount to Delta's local Lustre scratch, so
# training reads LMDB pages from local disk rather than over NFS.
#
# Why: the two constant-rate controls that read the build through Taiga (21934082,
# 21934084) both died mid-epoch on an NCCL all-reduce watchdog timeout (1,800 s) when one
# rank stopped answering; the only remote I/O in the training loop is the LMDB read
# through that mount. Unverified as the cause, but removing it costs 554 GB of a 9.8 TB
# quota with 7 TB free. Delta scratch holds the 010 build the same way.
#
# Two rsync passes: the second must transfer nothing (the same acceptance the Taiga
# transfer used, [[experiments.025-solid-growth.scripts.sync_taiga_025_build]]). The
# symlink swap at the end is manual and happens only after this job prints CLEAN:
#
#   B=/scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/025-solid-growth
#   mv $B/001-full-build $B/001-full-build.taiga-link && mv $B/001-full-build.local $B/001-full-build
#
#   sbatch experiments/025-solid-growth/scripts/delta_copy_025_build_local.slurm
#SBATCH --partition=cpu
#SBATCH --account=bfjt-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=12:00:00
#SBATCH --job-name=025-copy-build-local
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out

SRC=/taiga/illinois/eng/chbe/zhao5/mjvolk3/projects/torchcell/data/torchcell/experiments/025-solid-growth/001-full-build
DST=/scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/025-solid-growth/001-full-build.local

echo "job $SLURM_JOB_ID host $(hostname) start $(date)"
[[ -d "$SRC/processed/lmdb" ]] || { echo "ERROR: source missing $SRC/processed/lmdb" >&2; exit 1; }
mkdir -p "$DST"
df -h /scratch | tail -1

for pass in 1 2; do
  echo "=== rsync pass $pass $(date)"
  rsync -a --no-g --chmod=ug+rwX --info=stats2,progress2 "$SRC/" "$DST/" 2>&1 | tail -20
  echo "pass $pass exit ${PIPESTATUS[0]} $(date)"
done

SRC_B=$(stat -c %s "$SRC/processed/lmdb/data.mdb")
DST_B=$(stat -c %s "$DST/processed/lmdb/data.mdb")
echo "data.mdb bytes: source $SRC_B local $DST_B"
if [[ "$SRC_B" == "$DST_B" && "$SRC_B" == "554075586560" ]]; then
  echo "CLEAN: local copy complete, sizes match; swap the symlink (see header)."
else
  echo "MISMATCH: do not swap." >&2; exit 1
fi
echo "end $(date)"
