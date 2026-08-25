#!/bin/bash
# scripts/wandb_sync_offline.sh
# [[scripts.wandb_sync_offline]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/wandb_sync_offline.sh
#
# ONE-SHOT sync of offline W&B runs on a cluster login node. Run by hand / on request.
# NEVER run this in a loop on IGB -- persistent background processes on a shared login node
# are exactly what the head-node policy forbids, and an unattended loop cannot be seen or
# reaped by the user.
#
# WHY WE WRITE OUR OWN MARKER
# ---------------------------
# `wandb sync` writes `run-<id>.wandb.synced` only SOMETIMES. Verified on IGB 2026-07-28:
# a run synced successfully (exit 0, "... done.") and still got no marker, even when
# `--mark-synced` was passed explicitly -- while `wandb sync --sync-all` on the same
# directory immediately afterwards reported "Nothing to sync", i.e. wandb knew the run was
# synced but had not recorded it anywhere we could observe.
#
# Consequence of trusting wandb's marker: those runs count as pending forever, every pass
# re-uploads them, and the pending total never falls no matter how many passes are run.
# That is precisely the symptom this script was written to fix.
#
# So we write `.tc-synced` ourselves, only on a zero exit, containing the timestamp and the
# resulting run URL. It is our bookkeeping, it is observable, and the pass is IDEMPOTENT:
# running it twice does nothing the second time.
#
# Usage (on the login node):
#   bash wandb_sync_offline.sh                # every offline run
#   bash wandb_sync_offline.sh 2317356        # only paths matching this pattern (e.g. job id)
#   bash wandb_sync_offline.sh '' --recheck   # ignore our markers and re-attempt everything
set -uo pipefail

ROOT="${WANDB_OFFLINE_ROOT:-$HOME/scratch/torchcell/wandb-experiments}"
LOG="${WANDB_SYNC_LOG:-$HOME/scratch/torchcell/wandb_sync.log}"
FILTER="${1:-}"
RECHECK="${2:-}"

# shellcheck disable=SC1091
source "$HOME/miniconda3/bin/activate" 2>/dev/null || true
conda activate torchcell 2>/dev/null || true

if [ -n "$FILTER" ]; then
  mapfile -t DIRS < <(find "$ROOT" -maxdepth 3 -path "*${FILTER}*" -name "offline-run-*" -type d 2>/dev/null | sort)
else
  mapfile -t DIRS < <(find "$ROOT" -maxdepth 3 -name "offline-run-*" -type d 2>/dev/null | sort)
fi

total=${#DIRS[@]}
echo "[$(date +'%F %T')] sync start  filter=${FILTER:-<all>}  candidates=$total" >>"$LOG"
printf 'candidates: %d\n' "$total"

ok=0; skip=0; fail=0
for d in "${DIRS[@]}"; do
  if [ -f "$d/.tc-synced" ] && [ "$RECHECK" != "--recheck" ]; then
    skip=$((skip + 1))
    continue
  fi
  out=$(wandb sync --no-include-synced "$d" 2>&1)
  rc=$?
  echo "$out" >>"$LOG"
  if [ $rc -eq 0 ]; then
    url=$(printf '%s' "$out" | grep -oE 'https://wandb\.ai/[^ ]+' | head -1)
    printf 'synced_at=%s\nurl=%s\n' "$(date -Is)" "${url:-unknown}" >"$d/.tc-synced"
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "[FAIL rc=$rc] $d" >>"$LOG"
  fi
done

msg="sync done: $ok synced, $skip already marked, $fail failed (of $total)"
echo "[$(date +'%F %T')] $msg" >>"$LOG"
printf '%s\n' "$msg"
