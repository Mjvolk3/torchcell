#!/bin/bash
# scripts/tc_bench_promote_prod
# [[scripts.tc_bench_promote_prod]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/tc_bench_promote_prod
#
# `make bench-promote-prod CONFIRM=1`: start PRODUCTION of the benchmark service on the
# exact image that is running on staging. Production is never built, so what was
# exercised on staging is what the public board runs.
#
#   1. Read the image tag staging's API container is running. No running staging, no
#      promotion.
#   2. Refuse a tag whose commit is not on origin/main: production runs landed code
#      only. FORCE=1 overrides.
#   3. Without CONFIRM=1, print the plan (staging tag, production's current tag) and
#      stop. Production is public, so the bare target changes nothing.
#   4. scripts/tc_bench_deploy.sh production <tag>, which asserts that /health on the
#      production port reports tier=production and build=<tag>.
#
# The site is separate: build it at the same commit with `make site-build TIER=prod`.
# DRY_RUN=1 prints the commands only (and needs STAGING_TAG=<tag>, since nothing runs).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tc_bench_deploy.sh"

if [ "${DRY_RUN:-0}" = "1" ]; then
    STAGING_TAG="${STAGING_TAG:?DRY_RUN=1 needs STAGING_TAG=<tag>}"
    PRODUCTION_TAG="(not read in a dry run)"
else
    STAGING_TAG="$(running_tag tc-bench-staging)"
    PRODUCTION_TAG="$(running_tag tc-bench-production)"
fi
if [ -z "$STAGING_TAG" ]; then
    echo "refusing: staging is not running, so there is no soaked image to promote." >&2
    echo "Run: make bench-redeploy" >&2
    exit 1
fi
if ! git -C "$REPO" merge-base --is-ancestor "$STAGING_TAG" origin/main 2>/dev/null; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "refusing: staging runs $STAGING_TAG, which is not a commit on origin/main." >&2
        echo "Land it and redeploy staging first, or FORCE=1 to promote unlanded code." >&2
        exit 1
    fi
    echo "WARNING: promoting $STAGING_TAG, which is not on origin/main (FORCE=1)."
fi

echo "staging runs:     $STAGING_TAG"
echo "production runs:  ${PRODUCTION_TAG:-nothing}"
if [ "${CONFIRM:-0}" != "1" ]; then
    echo "plan only. To start production on $STAGING_TAG: make bench-promote-prod CONFIRM=1"
    exit 0
fi

deploy production "$STAGING_TAG"
echo "production is on $STAGING_TAG, the image staging ran."
