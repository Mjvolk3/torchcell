#!/usr/bin/env bash
# experiments/019-simb-multimodal/scripts/igb_login_wandb_sync.sh
# [[experiments.019-simb-multimodal.scripts.igb_login_wandb_sync]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/igb_login_wandb_sync.sh
#
# Sync IGB offline W&B runs from a login node WITHOUT tripping the login-node limits.
#
# WHY THIS EXISTS. IGB compute nodes have no internet, so every run on mmli/cabbi writes
# `offline-run-*` directories and is synced later; the login node is the only place that
# can reach wandb.ai, which makes this one of the few things legitimately run there. But
# `wandb sync wandb/offline-run-*` hands EVERY run to ONE python process, and that process
# is what generates
#
#     Process: python ... CPU%: 1.6  Mem%: 6.9  Limits: %cpu: 15.0 %mem: 5.0
#
# from the Biocluster resource monitor. The activity is allowed; the shape of the command
# is not. Peak memory scales with how many runs are handed to a single invocation, so the
# fix is to hand it one at a time.
#
# WHAT THIS DOES DIFFERENTLY
#   * ONE `wandb sync` process per offline-run directory, so peak memory is one run's
#     worth rather than the whole backlog.
#   * `nice -n 19`, so the sync yields to anything interactive on a shared login node.
#   * Skips runs already marked synced, so re-running after an interruption is cheap and
#     the whole thing is resumable. This is what makes one-at-a-time affordable.
#   * A short pause between runs, which is what keeps the AVERAGE CPU under the 15% cap;
#     a tight loop of short processes can average as high as one long one.
#   * Refuses to run anything if the wandb CLI is missing, rather than silently doing
#     nothing and leaving runs unsynced.
#
# WHAT IT DOES NOT DO. It does not train, evaluate, or load datasets. Nothing in this
# script belongs on a compute node and nothing else belongs on a login node: for actual
# compute use `sbatch`, or `srun --pty /bin/bash` for an interactive shell.
# See https://help.igb.illinois.edu/Biocluster
#
# Usage, from a biologin node:
#   bash experiments/019-simb-multimodal/scripts/igb_login_wandb_sync.sh <wandb-base-dir>
#   bash .../igb_login_wandb_sync.sh /home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal
#
# Optional environment:
#   SYNC_PAUSE_S   seconds between runs (default 2)
#   SYNC_LIMIT     stop after this many runs this pass (default 0 = all)
#   DRY_RUN        1 to list what would be synced and exit

set -euo pipefail

BASE="${1:-}"
PAUSE="${SYNC_PAUSE_S:-2}"
LIMIT="${SYNC_LIMIT:-0}"

if [[ -z "$BASE" ]]; then
  echo "usage: $0 <wandb-base-dir>   (the directory CONTAINING wandb/offline-run-*)" >&2
  exit 2
fi
if [[ ! -d "$BASE/wandb" ]]; then
  echo "no $BASE/wandb directory; is that the WANDB_BASE the slurm script set?" >&2
  exit 2
fi
if ! command -v wandb >/dev/null 2>&1; then
  echo "wandb CLI not on PATH; activate the environment first" >&2
  exit 2
fi

# A login node is the RIGHT place for this and the wrong place for everything else, so say
# where we are rather than guessing. SLURM_JOB_ID is set inside a job; its absence plus a
# `biologin` hostname is the intended case.
echo "host: $(hostname)   base: $BASE"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "note: running inside SLURM job ${SLURM_JOB_ID}; compute nodes usually have no internet."
fi

mapfile -t RUNS < <(find "$BASE/wandb" -maxdepth 1 -type d -name 'offline-run-*' | sort)
echo "found ${#RUNS[@]} offline run directories"

pending=()
for run in "${RUNS[@]}"; do
  # wandb drops a `.wandb.synced` marker beside the run once it has been uploaded.
  if compgen -G "$run/*.wandb.synced" >/dev/null; then
    continue
  fi
  pending+=("$run")
done
echo "${#pending[@]} not yet synced"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%s\n' "${pending[@]}"
  exit 0
fi

n=0
for run in "${pending[@]}"; do
  if [[ "$LIMIT" -gt 0 && "$n" -ge "$LIMIT" ]]; then
    echo "stopping at SYNC_LIMIT=$LIMIT; re-run to continue"
    break
  fi
  n=$((n + 1))
  echo "[$n/${#pending[@]}] $(basename "$run")"
  # ONE run per process. `|| true` so a single corrupt run does not abort the backlog;
  # it stays unsynced and is retried on the next pass, which the marker check makes cheap.
  nice -n 19 wandb sync "$run" || echo "  FAILED, left for the next pass: $run"
  sleep "$PAUSE"
done

echo "done; re-run to pick up anything left."
