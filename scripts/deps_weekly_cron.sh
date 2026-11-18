#!/usr/bin/env bash
# scripts/deps_weekly_cron.sh
# [[scripts.deps_weekly_cron]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/deps_weekly_cron.sh
#
# The Sunday cron entry of the weekly dependency job (scripts/deps_weekly.py). The
# crontab line does not run this file from a checkout: it pipes origin/main's copy in,
#
#   git -C <primary> fetch -q origin && \
#     git -C <primary> show origin/main:scripts/deps_weekly_cron.sh | bash -s -- <primary>
#
# so neither this script nor the job depends on what the primary checkout has checked
# out (it can sit behind origin/main). This script then:
#   1. adds a fresh detached worktree of origin/main under
#      /scratch/tmp/torchcell-deps-weekly/wt-<YYYY.MM.DD>;
#   2. runs `scripts/deps_weekly.py run --land` in it with the torchcell conda Python;
#   3. removes the worktree, unless the job left a branch in it (see the end).
#
# Exit code: the job's (0 done, 1 the suite failed on the newest dependencies, 2 the
# job failed).

set -euo pipefail

# Everything runs inside main, called once at the bottom: bash reads its script from
# the crontab's pipe, so a function makes it read the whole file before anything runs,
# and `< /dev/null` stops a child (git, gh, uv, pytest) from reading the rest of the
# script as its input.
main() {
    local PRIMARY PY DATE WT status
    PRIMARY="${1:?usage: deps_weekly_cron.sh <primary checkout>}"
    PY="$HOME/miniconda3/envs/torchcell/bin/python"
    DATE="$(date +%Y.%m.%d)"
    WT="/scratch/tmp/torchcell-deps-weekly/wt-$DATE"

    echo "[deps-weekly-cron] $(date -Is) origin/main $(git -C "$PRIMARY" rev-parse --short origin/main) -> $WT"
    mkdir -p "$(dirname "$WT")"
    git -C "$PRIMARY" worktree add --detach "$WT" origin/main
    cd "$WT"

    status=0
    # scripts/cron_run.sh (PR #1168) records each cron run for the `make jobs` panel. It
    # is checked for in the fresh worktree, not assumed, because this script can reach
    # origin/main before that wrapper does; the job runs the same either way.
    if [ -f scripts/cron_run.sh ]; then
        bash scripts/cron_run.sh deps-weekly -- "$PY" scripts/deps_weekly.py run --land || status=$?
    else
        "$PY" scripts/deps_weekly.py run --land || status=$?
    fi

    cd /
    # A worktree still detached means the job made no branch (no change, a red suite, or a
    # failure before the commit): it is scratch and goes. A worktree on a branch holds the
    # job's PR; the merge-queue drainer removes it when it lands it, and a PR left open
    # needs it to be landed later, so it stays. It can already be gone when the drainer
    # landed the branch while the job was finishing.
    if [ -d "$WT" ]; then
        if [ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = "HEAD" ]; then
            git -C "$PRIMARY" worktree remove --force "$WT"
            echo "[deps-weekly-cron] removed $WT"
        else
            echo "[deps-weekly-cron] kept $WT (branch $(git -C "$WT" rev-parse --abbrev-ref HEAD))"
        fi
    fi
    echo "[deps-weekly-cron] $(date -Is) exit $status"
    exit "$status"
}

main "$@" < /dev/null
