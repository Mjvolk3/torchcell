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

# Resubmit $1 (the slurm script path) iff enough time remains before the deadline.
#
# The resubmit's walltime is sized to the REMAINING window, not taken from the script's
# `#SBATCH --time`. This matters because the scripts request a multi-day walltime: SLURM
# refuses (or holds) a job whose time limit cannot fit before `--deadline`, so a fixed
# 4-day request resubmitted with 20h left would be REJECTED and the chain would die
# silently well before the deadline. Passing `--time=<remaining minutes>` keeps every
# resubmit valid right up to the end, and the last one simply runs out the clock.
requeue_if_before_deadline() {
  local script="$1"
  local now deadline_epoch remaining_min
  now=$(date +%s)
  deadline_epoch=$(_requeue_deadline_epoch) || return 0
  remaining_min=$(( (deadline_epoch - now) / 60 ))
  # Below this, a fresh job cannot import torch, build the dataset, and finish a trial --
  # it would only add a RUNNING row to the study. End the chain instead.
  if [ "$remaining_min" -lt "${REQUEUE_MIN_MINUTES:-45}" ]; then
    echo "[requeue] $(date -Is): only ${remaining_min}m left before $DEADLINE -- chain ends here."
    return 0
  fi
  # QUEUE SLACK -- do NOT request the full remaining window. SLURM refuses (State=DEADLINE,
  # Elapsed=00:00:00) any job whose TimeLimit cannot fit before --deadline, and it re-checks
  # at SCHEDULING time, not just submission. Requesting exactly `remaining` therefore only
  # works if the job starts INSTANTLY: a job that waits for busy GPUs is killed before it
  # ever runs. (Observed: the mmli 4-GPU morph arm died this way while the 3-GPU GH and
  # 2-GPU cabbi arms, which started immediately, survived.) Reserving slack lets a job sit
  # in the queue that long and still fit.
  local slack="${REQUEUE_QUEUE_SLACK_MINUTES:-120}"
  local req_min=$(( remaining_min - slack ))
  if [ "$req_min" -lt "${REQUEUE_MIN_MINUTES:-45}" ]; then
    req_min="${REQUEUE_MIN_MINUTES:-45}"
  fi
  echo "[requeue] $(date -Is): ${remaining_min}m left before $DEADLINE -- resubmitting $script (--time=${req_min}m, ${slack}m queue slack)"
  # A bare integer --time is MINUTES in SLURM. --deadline is kept as a belt-and-braces
  # backstop so the chain can never outlive the deadline.
  sbatch --time="$req_min" --deadline="$DEADLINE" "$script"
}

# Submit $1 sized to the remaining window before $DEADLINE. Use this for the FIRST
# submission of a chain, so the initial job is subject to exactly the same time arithmetic
# as a requeue: with a 5-day partition cap and a deadline under 5 days out, ONE job covers
# the whole window and the chain needs no handoff at all (a handoff costs one in-flight
# trial per worker, since Optuna resumes at trial granularity, not epoch).
#
#   DEADLINE="2026-07-29T10:00:00" \
#     bash experiments/019-simb-multimodal/scripts/requeue_until.sh submit <script>
submit_until_deadline() {
  requeue_if_before_deadline "$1"
}

# Allow `bash requeue_until.sh submit <script>` as a launcher, while still being sourceable.
if [ "${1:-}" = "submit" ] && [ -n "${2:-}" ]; then
  submit_until_deadline "$2"
fi

# Arm a SIGUSR1 trap that resubmits before the walltime kills this job.
requeue_arm_trap() {
  local script="$1"
  # shellcheck disable=SC2064  # intentional: expand $script now, at trap-arm time
  trap "echo '[requeue] caught SIGUSR1 (walltime approaching)'; requeue_if_before_deadline '$script'" USR1
}
