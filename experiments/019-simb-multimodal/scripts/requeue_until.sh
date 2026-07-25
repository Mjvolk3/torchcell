#!/bin/bash
# experiments/019-simb-multimodal/scripts/requeue_until.sh
# [[experiments.019-simb-multimodal.scripts.requeue_until]]
# Deadline-guarded SLURM self-resubmit helper for the _003 decoder sweeps.
#
# WHY: partition walltimes (20h-3d) are shorter than the time we want the sweep to run
# (until a fixed wall-clock deadline). Optuna's SQLite study PERSISTS across jobs, so a
# resubmitted job's workers simply reattach to the same study and keep adding trials --
# the search grows monotonically until the deadline instead of being capped by one job's
# walltime. Fresh `_003` studies are used because the categorical space changed (adding
# `decoder`/`dist` to an existing study would corrupt the TPE model's assumptions).
#
# USAGE (source this from a slurm script, BEFORE launching the workers):
#
#   DEADLINE="2026-07-29T10:00:00"
#   source "$(dirname "$0")/requeue_until.sh"
#   requeue_arm_trap "$0"            # resubmit on SIGUSR1 (sent by --signal=B:USR1@180)
#   ... launch workers ... ; wait
#   requeue_if_before_deadline "$0"  # also resubmit on a CLEAN exit before the deadline
#
# The slurm script must set:
#   #SBATCH --signal=B:USR1@180   -> SIGUSR1 to the BATCH shell 180s before the walltime
# and should set --deadline so SLURM itself refuses to start a job that cannot finish.
#
# NOTE: `--signal=B:...` requires the batch script to be running the wait loop in the
# foreground (`wait`), which all three launchers do.

# Resolve the deadline (env DEADLINE, ISO-8601 local time) to an epoch once.
_requeue_deadline_epoch() {
  if [ -z "$DEADLINE" ]; then
    echo "[requeue] DEADLINE unset -- refusing to resubmit" >&2
    return 1
  fi
  date -d "$DEADLINE" +%s
}

# Resubmit $1 (the slurm script path) iff we are still before the deadline.
requeue_if_before_deadline() {
  local script="$1"
  local now deadline_epoch
  now=$(date +%s)
  deadline_epoch=$(_requeue_deadline_epoch) || return 0
  if [ "$now" -lt "$deadline_epoch" ]; then
    echo "[requeue] $(date -Is): before deadline $DEADLINE -- resubmitting $script"
    # --deadline makes SLURM itself refuse to start a job that would run past the deadline,
    # so the chain terminates cleanly even if this guard is somehow reached late.
    sbatch --deadline="$DEADLINE" "$script"
  else
    echo "[requeue] $(date -Is): deadline $DEADLINE passed -- chain ends here."
  fi
}

# Arm a SIGUSR1 trap that resubmits before the walltime kills this job.
requeue_arm_trap() {
  local script="$1"
  # shellcheck disable=SC2064  # intentional: expand $script now, at trap-arm time
  trap "echo '[requeue] caught SIGUSR1 (walltime approaching)'; requeue_if_before_deadline '$script'" USR1
}
