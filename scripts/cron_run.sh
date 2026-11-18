#!/usr/bin/env bash
# scripts/cron_run.sh
# [[scripts.cron_run]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/cron_run.sh
#
# Run one recurring job and record how it went, so `make jobs` can say whether the
# last run of each cron job succeeded.
#
#   bash scripts/cron_run.sh <job-name> -- <command...>
#
# The command's stdout and stderr are appended to the job's log,
#   ${TORCHCELL_CRON_LOG_DIR:-/tmp}/torchcell-<job-name>.log
# (the same /tmp/torchcell-*.log names the crontab used before the wrapper), between
# a start and an end marker line. Two JSON records go to the state directory
#   ${XDG_STATE_HOME:-$HOME/.local/state}/torchcell/cron/
# <job>.json is the latest run: written when the job starts (finished_at and
# exit_code null, so a run in progress reads as running) and rewritten when it ends.
# <job>.history.jsonl gets one line per finished run, capped at the newest
# TORCHCELL_CRON_HISTORY_MAX lines (default 500: over a year of a daily job).
# Each record: job, started_at, finished_at, exit_code, host, log. Writes go to a temp
# file in the same directory and are renamed into place, under a per-job flock, so a
# reader never sees half a record.
#
# The job name is also the registry key: scripts/jobs.py holds the registry (name,
# schedule, command) and tests/scripts/test_jobs.py checks every wrapped line of
# scripts/crontab.txt against it.
#
# Exit code: the command's own, so cron's mail and any caller see what they saw before.

set -uo pipefail

usage() {
    echo "usage: $0 <job-name> -- <command...>" >&2
    exit 2
}

[[ $# -ge 3 && "$2" == "--" ]] || usage
JOB="$1"
shift 2
[[ "$JOB" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "cron_run: bad job name '$JOB' (want [a-z0-9-])" >&2; exit 2; }

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/torchcell/cron"
LOG="${TORCHCELL_CRON_LOG_DIR:-/tmp}/torchcell-${JOB}.log"
HISTORY_MAX="${TORCHCELL_CRON_HISTORY_MAX:-500}"
HOST="$(hostname -s)"
# The records are built with printf, so a path that needs JSON escaping is refused.
[[ "$LOG" != *[\"\\]* ]] || { echo "cron_run: log path '$LOG' contains a quote or backslash" >&2; exit 2; }
mkdir -p "$STATE_DIR"

record() {  # started finished exit -> one JSON object (finished/exit may be null)
    printf '{"job": "%s", "started_at": "%s", "finished_at": %s, "exit_code": %s, "host": "%s", "log": "%s"}\n' \
        "$JOB" "$1" "$2" "$3" "$HOST" "$LOG"
}

# write_latest <json> [append-to-history]
write_latest() {
    (
        flock 9
        local tmp
        tmp="$(mktemp "$STATE_DIR/.${JOB}.json.XXXXXX")"
        printf '%s\n' "$1" >"$tmp"
        mv -f "$tmp" "$STATE_DIR/${JOB}.json"
        if [[ "${2:-}" == "history" ]]; then
            local hist="$STATE_DIR/${JOB}.history.jsonl"
            tmp="$(mktemp "$STATE_DIR/.${JOB}.history.XXXXXX")"
            { [[ -f "$hist" ]] && cat "$hist"; printf '%s\n' "$1"; } | tail -n "$HISTORY_MAX" >"$tmp"
            mv -f "$tmp" "$hist"
        fi
    ) 9>"$STATE_DIR/${JOB}.lock"
}

STARTED="$(date -Iseconds)"
write_latest "$(record "$STARTED" null null)"
echo "=== cron_run ${JOB} start ${STARTED} on ${HOST}: $* ===" >>"$LOG"

"$@" >>"$LOG" 2>&1
RC=$?

FINISHED="$(date -Iseconds)"
echo "=== cron_run ${JOB} end ${FINISHED} exit ${RC} ===" >>"$LOG"
write_latest "$(record "$STARTED" "\"$FINISHED\"" "$RC")" history
exit "$RC"
