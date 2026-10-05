#!/bin/bash
# scripts/tc_bench_deploy
# [[scripts.tc_bench_deploy]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/tc_bench_deploy
#
# The deploy primitive of the benchmark service. Starts one tier (staging or
# production) of docker-compose.tc-bench.yml on one image tag, then checks that the
# tier answers on its port with that tier name and that build.
#
#   scripts/tc_bench_deploy.sh staging            build the image from the current tree,
#                                                 tagged with HEAD's short sha, and start
#                                                 staging on it
#   scripts/tc_bench_deploy.sh production <tag>   start production on an image that
#                                                 already exists; never builds
#
# Normal use is through the two wrappers, which add the preflight and the promotion
# rule: `make bench-redeploy` (scripts/tc_bench_redeploy.sh) and
# `make bench-promote-prod CONFIRM=1` (scripts/tc_bench_promote_prod.sh).
#
# DRY_RUN=1 prints every docker and curl command instead of running it.
# Functions are shared with the wrappers: they `source` this file.

set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
BASE_FILE="$REPO/docker-compose.tc-bench.yml"
IMAGE="tc-bench-api"
API_SERVICE="tc-bench-api"

# Prints the command when DRY_RUN=1, runs it otherwise.
run() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf 'DRY_RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# tier_setup <staging|production>: sets PROJECT, ENV_FILE, OVERRIDE and PORT.
tier_setup() {
    case "$1" in
        staging)
            PROJECT="tc-bench-staging"
            ENV_FILE="$REPO/.env.tc-bench.staging"
            OVERRIDE="$REPO/docker-compose.tc-bench.staging.yml"
            ;;
        production)
            PROJECT="tc-bench-production"
            ENV_FILE="$REPO/.env.tc-bench.prod"
            OVERRIDE="$REPO/docker-compose.tc-bench.prod.yml"
            ;;
        *)
            echo "tier must be staging or production, got: $1" >&2
            exit 2
            ;;
    esac
    if [ ! -f "$ENV_FILE" ]; then
        echo "missing $ENV_FILE (copy docker/tc-bench/tc-bench.<tier>.env.example)" >&2
        exit 2
    fi
    PORT="$(sed -n 's/^TC_BENCH_PORT=//p' "$ENV_FILE")"
    if [ -z "$PORT" ]; then
        echo "$ENV_FILE does not set TC_BENCH_PORT" >&2
        exit 2
    fi
}

# compose <args...>: docker compose for the tier chosen by tier_setup, on $TAG.
compose() {
    run env TC_BENCH_IMAGE_TAG="$TAG" docker compose -p "$PROJECT" \
        --env-file "$ENV_FILE" -f "$BASE_FILE" -f "$OVERRIDE" "$@"
}

# running_tag <project>: the image tag the project's API container runs, or nothing.
running_tag() {
    local container
    container="$(docker compose -p "$1" ps -q "$API_SERVICE" 2>/dev/null | head -n 1)"
    if [ -n "$container" ]; then
        docker inspect --format '{{.Config.Image}}' "$container" | sed 's/^.*://'
    fi
}

# assert_live <tier> <tag>: /health on the tier's port reports that tier and build.
assert_live() {
    local tier="$1" tag="$2" url="http://127.0.0.1:$PORT/api/v1/health" body=""
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN: curl $url   # expect tier=$tier build=$tag"
        return 0
    fi
    for _ in $(seq 1 30); do
        body="$(curl -fsS --max-time 3 "$url" 2>/dev/null || true)"
        if [ -n "$body" ]; then
            break
        fi
        sleep 2
    done
    if [ -z "$body" ]; then
        echo "FAILED: $tier did not answer on $url" >&2
        exit 1
    fi
    python3 - "$tier" "$tag" "$body" <<'PY'
import json
import sys

tier, tag, body = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
if (body["status"], body["tier"], body["build"]) != ("ok", tier, tag):
    sys.exit(f"FAILED: expected tier={tier} build={tag}, the API reports {body}")
print(f"live: {tier} on build {tag}, {body['n_datasets']} dataset(s)")
PY
}

# deploy <tier> [tag]: the whole primitive.
deploy() {
    local tier="$1"
    tier_setup "$tier"
    if [ "$tier" = "staging" ] && [ $# -eq 1 ]; then
        TAG="$(git -C "$REPO" rev-parse --short=9 HEAD)"
        compose build "$API_SERVICE"
    elif [ $# -eq 2 ]; then
        TAG="$2"
        if [ "${DRY_RUN:-0}" != "1" ] && ! docker image inspect "$IMAGE:$TAG" >/dev/null 2>&1; then
            echo "image $IMAGE:$TAG does not exist; production never builds" >&2
            exit 1
        fi
    else
        echo "usage: tc_bench_deploy.sh staging | tc_bench_deploy.sh <tier> <tag>" >&2
        exit 2
    fi
    compose up -d --no-build
    assert_live "$tier" "$TAG"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    deploy "$@"
fi
