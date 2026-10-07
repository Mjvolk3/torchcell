#!/bin/bash
# scripts/tc_site_publish
# [[scripts.tc_site_publish]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/tc_site_publish
#
# `make site-publish TIER=staging|prod`: build one tier's website from
# website/site.<tier>.env and copy it into the directory the reverse proxy serves for
# that tier (SITE_PUBLISH_DIR in the same env file; docker/tc-proxy/Caddyfile). This is
# the site half of a staging redeploy; `make bench-redeploy` is the API half.
#
#   1. The build tree. By default the build runs in website/ itself. On a host where
#      node_modules must stay off the repository disk (Radiant: 40 GB root disk), set
#      TC_SITE_BUILD_DIR to a prepared copy that already holds node_modules; the website
#      sources are rsynced into it first (node_modules, build outputs and .docusaurus
#      excluded), as the scratch preview scripts do.
#   2. `docusaurus build --out-dir build-<tier>` with the tier's env file exported.
#   3. A staging build must carry the noindex tag SITE_ENV=staging adds; the publish
#      refuses otherwise, so a production-looking build cannot land on the staging host.
#   4. rsync into SITE_PUBLISH_DIR with --delete, so removed pages disappear too.
#
# A dirty tree is allowed: staging is where unlanded work is looked at. The commit and
# the dirty flag are printed so the deployed site can be named.

set -euo pipefail

TIER="${1:-}"
case "$TIER" in
    staging|prod) ;;
    *) echo "usage: tc_site_publish.sh staging|prod" >&2; exit 2 ;;
esac

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
ENV_FILE="$REPO/website/site.$TIER.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "missing $ENV_FILE (copy website/site.$TIER.env.example)" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
: "${SITE_PUBLISH_DIR:?$ENV_FILE does not set SITE_PUBLISH_DIR}"
: "${SITE_ENV:?$ENV_FILE does not set SITE_ENV}"

BUILD_DIR="${TC_SITE_BUILD_DIR:-$REPO/website}"
if [ "$BUILD_DIR" != "$REPO/website" ]; then
    mkdir -p "$BUILD_DIR"
    rsync -a --exclude node_modules --exclude 'build*' --exclude .docusaurus \
        "$REPO/website/" "$BUILD_DIR/"
fi
if [ ! -d "$BUILD_DIR/node_modules" ]; then
    echo "no node_modules in $BUILD_DIR; run npm ci there first" >&2
    exit 1
fi

OUT="$BUILD_DIR/build-$TIER"
(cd "$BUILD_DIR" && nice -n 15 npx docusaurus build --out-dir "$OUT")
test -s "$OUT/index.html"

if [ "$TIER" = "staging" ] && ! grep -q 'noindex' "$OUT/index.html"; then
    echo "refusing: the staging build has no noindex tag (SITE_ENV=$SITE_ENV); a staging" >&2
    echo "site must be built with SITE_ENV=staging" >&2
    exit 1
fi

# World-readable on purpose: the proxy container reads the site as a user with no
# override capability, and a build tree can carry a group-only mode (seen 2026-10-07:
# 660 files from the scratch copy gave 403 on every page).
mkdir -p "$SITE_PUBLISH_DIR"
rsync -a --delete --chmod=D755,F644 "$OUT/" "$SITE_PUBLISH_DIR/"

COMMIT="$(git -C "$REPO" rev-parse --short=9 HEAD)"
DIRTY=""
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    DIRTY=" (dirty tree)"
fi
echo "published $TIER site from $COMMIT$DIRTY to $SITE_PUBLISH_DIR at $(date -Is)"
