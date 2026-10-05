#!/bin/bash
# scripts/tc_bench_redeploy
# [[scripts.tc_bench_redeploy]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/tc_bench_redeploy
#
# `make bench-redeploy`: rebuild STAGING of the benchmark service from the current tree
# and prove it is live. Staging is the only tier that is ever built.
#
#   1. Preflight. A dirty tree is refused, because the image is tagged with HEAD's sha
#      and a dirty build would carry that tag without being that commit. A HEAD that is
#      not on origin/main is allowed (staging exists to try things) but said out loud.
#   2. scripts/tc_bench_deploy.sh staging: build, start, and assert that /health on the
#      staging port reports tier=staging and build=<HEAD short sha>.
#
# FORCE=1 skips the dirty-tree refusal. DRY_RUN=1 prints the commands only.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tc_bench_deploy.sh"

if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "refusing: the working tree has uncommitted changes, so the image would not" >&2
        echo "be the commit its tag names. Commit them, or FORCE=1 to build anyway." >&2
        exit 1
    fi
    echo "WARNING: building from a dirty tree (FORCE=1); the tag will not name this build exactly."
fi
if ! git -C "$REPO" merge-base --is-ancestor HEAD origin/main 2>/dev/null; then
    echo "note: HEAD ($(git -C "$REPO" rev-parse --short=9 HEAD)) is not on origin/main; staging will run unlanded code."
fi

deploy staging
echo "staging is on $TAG. Promote it with: make bench-promote-prod CONFIRM=1"
