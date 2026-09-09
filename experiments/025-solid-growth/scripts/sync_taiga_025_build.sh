#!/bin/bash
# experiments/025-solid-growth/scripts/sync_taiga_025_build.sh
# [[experiments.025-solid-growth.scripts.sync_taiga_025_build]]
#
# Mirror the 025 all-solid-growth build from GilaHyper onto the Zhao-group Taiga share, by way
# of the Radiant VM, so Delta can read it WITHOUT a Duo prompt:
#
#   GilaHyper --rsync over ssh (key auth)--> rocky@torchcell-database:/mnt/zhao5/...
#   Delta login/compute nodes see the same bytes at /taiga/illinois/eng/chbe/zhao5/...
#
# Same shape as sync_igb_025_build.sh: only processed/ (554,075,586,560 B LMDB) and
# data_module_cache/ go, plus the empty raw/lmdb stub PyG's _download looks for. The subset
# and split artifacts arrive by git.
#
#   bash experiments/025-solid-growth/scripts/sync_taiga_025_build.sh
#   DRY_RUN=1 bash .../sync_taiga_025_build.sh
#
# Resumable (rsync -aP, no --delete). The Taiga share is exported to Radiant as the LOWERCASE
# path /taiga/illinois/eng/chbe/zhao5 (NCSA SUP-29573), mounted at /mnt/zhao5; `rocky` writes
# through group 555647, so files land as uid 1000 gid 555647 with group read, which is what
# mjvolk3 on Delta reads with.
set -euo pipefail

GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
REL="data/torchcell/experiments/025-solid-growth/001-full-build"
SRC="$GH_DATA_ROOT/$REL"

RADIANT="${RADIANT:-rocky@141.142.216.218}"
# Delta sees this as /taiga/illinois/eng/chbe/zhao5/mjvolk3/projects/torchcell/$REL
TAIGA_ROOT="${TAIGA_ROOT:-/mnt/zhao5/mjvolk3/projects/torchcell}"
DEST_DIR="$TAIGA_ROOT/$REL"

for d in processed data_module_cache; do
  [[ -d "$SRC/$d" ]] || { echo "ERROR: source not found: $SRC/$d" >&2; exit 1; }
done

echo "== 025 build sync GilaHyper -> Taiga (via Radiant) =="
echo "  src : $SRC"
echo "  dest: $RADIANT:$DEST_DIR"
echo "  delta sees: /taiga/illinois/eng/chbe/zhao5/mjvolk3/projects/torchcell/$REL"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "== DRY_RUN: nothing sent =="
  exit 0
fi

CTL="${TMPDIR:-/tmp}/radiant-025-sync-%r@%h:%p"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=4h -o BatchMode=yes)
trap 'ssh "${SSH_OPTS[@]}" -O exit "$RADIANT" 2>/dev/null || true' EXIT

# GROUP ACCESS, twice over. The NFS server resolves rocky's groups itself and does not see
# the local zhao5_nfs membership, so a plain ssh session is refused at /mnt/zhao5; only a
# process whose PRIMARY group is 555647 (`sg zhao5_nfs`) gets in. Hence every remote command
# runs under sg, and the remote rsync is /usr/local/bin/rsync-zhao5, a two-line wrapper
# (`exec sg zhao5_nfs -c "rsync $*"`) installed on the VM 2026-09-09. Then the files must
# be GROUP-readable, since Delta reads them as mjvolk3 (uid 67392) through gid 555647, not
# as uid 1000: --chmod=ug+rwX. --no-g skips preserving GilaHyper's gid, which the VM cannot
# set anyway (rsync exit 23 noise); the setgid directory assigns 555647.
RSYNC_REMOTE=/usr/local/bin/rsync-zhao5
remote() { ssh "${SSH_OPTS[@]}" "$RADIANT" "sg zhao5_nfs -c '$1'"; }

remote "mkdir -p $DEST_DIR/raw/lmdb && touch $DEST_DIR/.write_test && rm -f $DEST_DIR/.write_test"

for d in processed data_module_cache; do
  echo "== rsync $d  $(date) =="
  rsync -aP --no-g --chmod=ug+rwX --human-readable --info=progress2 \
    --rsync-path="$RSYNC_REMOTE" -e "ssh ${SSH_OPTS[*]}" \
    "$SRC/$d/" "$RADIANT:$DEST_DIR/$d/"
done

echo "== verifying: a second pass must report no transfers  $(date) =="
rsync -a --no-g --chmod=ug+rwX --itemize-changes --dry-run \
  --rsync-path="$RSYNC_REMOTE" -e "ssh ${SSH_OPTS[*]}" \
  "$SRC/processed/" "$RADIANT:$DEST_DIR/processed/" | head -20
remote "ls -ln $DEST_DIR/processed/lmdb; df -h $DEST_DIR | tail -1"

echo "== done  $(date) =="
