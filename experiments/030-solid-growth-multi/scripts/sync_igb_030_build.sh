#!/bin/bash
# experiments/030-solid-growth-multi/scripts/sync_igb_030_build.sh
# [[experiments.030-solid-growth-multi.scripts.sync_igb_030_build]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/sync_igb_030_build.sh
#
# WHY. Mirror the 030 multi-measurement build (001-multi-build, 13.5M records) from
# GilaHyper to IGB Biocluster scratch so the OFFLINE mmli compute nodes can train the
# per-entry dataset-token arms with no Neo4j in reach. Same shape as
# experiments/029-solid-growth-ko/scripts/sync_igb_029_build.sh. Three things cross:
#
#   processed/          the only stage a loader reads: lmdb/data.mdb (955,268,878,336
#                       bytes) and lock.mdb, the index JSONs (dataset_name_index,
#                       perturbation_count_index, phenotype_label_index), label_df.parquet,
#                       experiment_types.json, gene_set.json, pre_filter.pt,
#                       pre_transform.pt, STAGE_COMPLETE and the zero-byte .lock files
#   raw/lmdb/           an EMPTY stub: PyG's files_exist check needs the directory, and the
#                       compute node must never try to rebuild from a Neo4j it cannot reach
#   data_module_cache/  the split indices (index_seed_*.json + index_details_seed_*.json)
#                       written by the GilaHyper smoke run. igb_mmli_cgt_030.slurm REFUSES
#                       to start without them: the datamodule's cache check is a bare
#                       osp.exists with no lock, so four DDP ranks would each scan 13.5M
#                       records to compute the same split. When the directory does not
#                       exist yet (first run, before the smoke job) the script prints a
#                       loud MISSING banner and still sends the rest; rerun after the smoke
#                       job and rsync sends only the cache.
#
# On GilaHyper $DATA_ROOT/$REL is a SYMLINK to
# /db/experiments/030-solid-growth-multi-001-multi-build (/db is at 90% and is read only
# here; nothing is staged on it). The script resolves the link with readlink -f and
# rsyncs the resolved directories, so the contents cross, not the link.
#
# RUN FROM GilaHyper, normally under gh_sync_igb_030.slurm (2 days, 8 GB). Resumable:
# rsync -a --partial --append-verify, no --delete, so a killed job is resubmitted, not
# repaired. --append-verify assumes the source is immutable, which a finished build is: it
# appends to a shorter destination file and checksums the whole file afterwards, falling
# back to a full re-send on mismatch. Bandwidth is not limited. The 029 sync moved
# 587.52 GB in 1 h 25 m, so expect 2.5 to 3.5 h for the 955 GB here.
#
#   bash experiments/030-solid-growth-multi/scripts/sync_igb_030_build.sh
#   DRY_RUN=1 bash experiments/030-solid-growth-multi/scripts/sync_igb_030_build.sh   # size it, send nothing
#
# At the end it prints local vs remote file counts, du -s of processed/ and the byte size
# of data.mdb (remote side via one ssh call), and exits 1 when the counts or the data.mdb
# bytes differ. du -s can differ by a few blocks across filesystems; the byte size and the
# file count are the exact comparison.
set -euo pipefail

GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
REL="data/torchcell/experiments/030-solid-growth-multi/001-multi-build"
SRC="$GH_DATA_ROOT/$REL"

IGB_USER="${IGB_USER:-mjvolk3}"
IGB_HOST="${IGB_HOST:-biologin.igb.illinois.edu}"
IGB_DATA_ROOT="${IGB_DATA_ROOT:-/home/a-m/mjvolk3/scratch/torchcell}"
DEST_DIR="$IGB_DATA_ROOT/$REL"

if [[ ! -d "$SRC/processed" ]]; then
  echo "ERROR: source not found: $SRC/processed" >&2
  exit 1
fi
# Follow the symlink: rsync of "$SRC_REAL/processed/" copies contents, never the link.
SRC_REAL="$(readlink -f "$SRC")"
if [[ ! -f "$SRC_REAL/processed/lmdb/data.mdb" ]]; then
  echo "ERROR: no LMDB at $SRC_REAL/processed/lmdb/data.mdb" >&2
  exit 1
fi

HAVE_CACHE=0
if [[ -d "$SRC_REAL/data_module_cache" ]]; then
  HAVE_CACHE=1
  n_idx=$(find "$SRC_REAL/data_module_cache" -maxdepth 1 -name 'index_seed_*.json' | wc -l)
  if [[ "$n_idx" -eq 0 ]]; then
    echo "WARNING: $SRC_REAL/data_module_cache exists but holds no index_seed_*.json;" >&2
    echo "         the launcher preflight will refuse to start until the smoke job writes one." >&2
  fi
fi

echo "== 030 build sync GilaHyper -> IGB =="
echo "  src : $SRC -> $SRC_REAL"
echo "        processed  $(du -sh "$SRC_REAL/processed" | cut -f1)"
echo "        data.mdb   $(stat -c %s "$SRC_REAL/processed/lmdb/data.mdb") bytes"
if [[ "$HAVE_CACHE" -eq 1 ]]; then
  echo "        data_module_cache  $(du -sh "$SRC_REAL/data_module_cache" | cut -f1), $(find "$SRC_REAL/data_module_cache" -maxdepth 1 -type f | wc -l) files"
else
  echo "        data_module_cache  ABSENT"
fi
echo "  dest: $IGB_USER@$IGB_HOST:$DEST_DIR"

if [[ "$HAVE_CACHE" -eq 0 ]]; then
  cat >&2 <<'EOF'
##########################################################################
#  MISSING: data_module_cache/ does not exist under the 030 build root.  #
#  processed/ and the raw/lmdb stub are sent anyway, but                 #
#  igb_mmli_cgt_030.slurm will REFUSE to start until the split cache     #
#  (index_seed_*.json) is written by the GilaHyper smoke job and this    #
#  script is rerun. The rerun sends only the cache.                      #
##########################################################################
EOF
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "== DRY_RUN: nothing sent, no SSH opened =="
  exit 0
fi

CTL="${TMPDIR:-/tmp}/igb-030-sync-%r@%h:%p"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=4h)
trap 'ssh "${SSH_OPTS[@]}" -O exit "$IGB_USER@$IGB_HOST" 2>/dev/null || true' EXIT

# The empty raw/lmdb stub is created on the far side; nothing under raw/ is sent.
ssh "${SSH_OPTS[@]}" "$IGB_USER@$IGB_HOST" "mkdir -p '$DEST_DIR/raw/lmdb' '$DEST_DIR/processed'"

RSYNC_OPTS=(-a --partial --append-verify --human-readable --info=progress2)

echo "== rsync processed =="
rsync "${RSYNC_OPTS[@]}" \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SRC_REAL/processed/" "$IGB_USER@$IGB_HOST:$DEST_DIR/processed/"

if [[ "$HAVE_CACHE" -eq 1 ]]; then
  echo "== rsync data_module_cache =="
  rsync "${RSYNC_OPTS[@]}" \
    -e "ssh ${SSH_OPTS[*]}" \
    "$SRC_REAL/data_module_cache/" "$IGB_USER@$IGB_HOST:$DEST_DIR/data_module_cache/"
fi

echo "== verifying: a second pass must report no file transfers =="
# Written to a file, not piped into head: under pipefail a long itemize list would give
# rsync a SIGPIPE and set -e would abort before the counts below.
ITEMIZE="${TMPDIR:-/tmp}/igb-030-sync-itemize.$$"
rsync -a --itemize-changes --dry-run -e "ssh ${SSH_OPTS[*]}" \
  "$SRC_REAL/processed/" "$IGB_USER@$IGB_HOST:$DEST_DIR/processed/" > "$ITEMIZE"
head -20 "$ITEMIZE"
PENDING=$(grep -c '^>f' "$ITEMIZE" || true)
echo "  pending file transfers: $PENDING"
rm -f "$ITEMIZE"

echo "== verifying: counts and sizes, local vs remote =="
LOCAL_N=$(find "$SRC_REAL/processed" -type f | wc -l)
LOCAL_DU=$(du -s "$SRC_REAL/processed" | cut -f1)
LOCAL_MDB=$(stat -c %s "$SRC_REAL/processed/lmdb/data.mdb")
LOCAL_CACHE_N=0
if [[ "$HAVE_CACHE" -eq 1 ]]; then
  LOCAL_CACHE_N=$(find "$SRC_REAL/data_module_cache" -type f | wc -l)
fi
# One ssh call returns four whitespace-separated numbers: files, du KiB, data.mdb bytes,
# cache files (0 when the directory is absent on IGB).
read -r REMOTE_N REMOTE_DU REMOTE_MDB REMOTE_CACHE_N < <(ssh "${SSH_OPTS[@]}" "$IGB_USER@$IGB_HOST" \
  "n=\$(find '$DEST_DIR/processed' -type f | wc -l); \
   d=\$(du -s '$DEST_DIR/processed' | cut -f1); \
   m=\$(stat -c %s '$DEST_DIR/processed/lmdb/data.mdb'); \
   c=0; [ -d '$DEST_DIR/data_module_cache' ] && c=\$(find '$DEST_DIR/data_module_cache' -type f | wc -l); \
   echo \"\$n \$d \$m \$c\"")

printf '  %-28s %20s %20s\n' "" "local (GilaHyper)" "remote (IGB)"
printf '  %-28s %20s %20s\n' "processed/ files" "$LOCAL_N" "$REMOTE_N"
printf '  %-28s %20s %20s\n' "processed/ du -s (KiB)" "$LOCAL_DU" "$REMOTE_DU"
printf '  %-28s %20s %20s\n' "processed/lmdb/data.mdb bytes" "$LOCAL_MDB" "$REMOTE_MDB"
printf '  %-28s %20s %20s\n' "data_module_cache/ files" "$LOCAL_CACHE_N" "$REMOTE_CACHE_N"

STATUS=0
[[ "$PENDING" -eq 0 ]] || { echo "MISMATCH: $PENDING file transfers still pending" >&2; STATUS=1; }
[[ "$LOCAL_N" == "$REMOTE_N" ]] || { echo "MISMATCH: processed/ file count" >&2; STATUS=1; }
[[ "$LOCAL_MDB" == "$REMOTE_MDB" ]] || { echo "MISMATCH: data.mdb byte size" >&2; STATUS=1; }
[[ "$LOCAL_CACHE_N" == "$REMOTE_CACHE_N" ]] || { echo "MISMATCH: data_module_cache/ file count" >&2; STATUS=1; }
[[ "$LOCAL_DU" == "$REMOTE_DU" ]] || echo "note: du -s differs; block accounting differs across filesystems, bytes and counts decide"

if [[ "$STATUS" -eq 0 ]]; then
  echo "== done: counts and data.mdb bytes match =="
else
  echo "== done WITH MISMATCHES: resubmit the sync (rsync resumes) =="
fi
echo "On IGB, DATA_ROOT=$IGB_DATA_ROOT and the loader root resolves to:"
echo "  $DEST_DIR"
exit "$STATUS"
