#!/bin/bash
# scripts/migrate_storage_tiers.sh
# [[scripts.migrate_storage_tiers]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/migrate_storage_tiers
#
# Populate the new tiers after setup_storage_tiers.sh has created /db and /bulk:
# move the neo4j serving tree to /db (SN850X), archive cold database bytes to /bulk
# (RAID1), and restart the read-only serving container from /db. Frees ~1.45 TB on
# /scratch, which becomes the ML tier.
#
#   sudo bash scripts/migrate_storage_tiers.sh                 # dry run
#   sudo bash scripts/migrate_storage_tiers.sh --apply         # copy + swap container
#   sudo bash scripts/migrate_storage_tiers.sh --apply --purge # also delete verified /scratch copies
#
# Copies are rsync -a and NON-DESTRUCTIVE; sources are deleted only under --purge,
# and only after a second rsync pass confirms zero remaining transfers. The serving
# container is stopped for the store copy (a live LMDB/neo4j store must never be
# copied hot) and restarted against /db at the end.
#
# WHAT GOES WHERE, and why:
#   /db    data/databases + data/transactions + data/dbms + server_id + cluster-state,
#          conf, logs, plugins, metrics, biocypher, .env -- everything the read-only
#          container mounts. All REGENERABLE (re-import from the archived CSV takes
#          ~19 min), which is the rule for the SN850X: no power-loss protection, so
#          nothing irreplaceable lives on it. Dataset LMDBs (data/torchcell, data/sgd,
#          data/string, data/tflink, data/go) are BUILD inputs, not serving inputs --
#          they stay on /scratch until the next build-tree decision.
#   /bulk  biocypher-out (both the radiant-backing 2026-07-22 tree and the live
#          build's CSV -- the re-import insurance for /db), data/dumps,
#          neo4j4-store-graveyard, /scratch/projects/torchcell-deprecated, and a
#          COPY (not move) of the torchcell-library mirror -- tc-lit-endpoint keeps
#          serving the original; the /bulk copy is the redundancy the mirror never had.
set -euo pipefail

SRC="${SRC:-/scratch/projects/torchcell/database}"
DEPRECATED_SRC="${DEPRECATED_SRC:-/scratch/projects/torchcell-deprecated}"
LIBRARY_SRC="${LIBRARY_SRC:-/scratch/projects/torchcell-scratch/torchcell-library}"
DB_ROOT="${DB_ROOT:-/db/database}"
BULK_ROOT="${BULK_ROOT:-/bulk}"
CONTAINER="${CONTAINER:-tc-neo4j-readonly}"
IMAGE="${IMAGE:-michaelvolk/tc-neo4j:5.26.28}"
SERVE_CPUS="${SERVE_CPUS:-16}"
SERVE_MEM="${SERVE_MEM:-200g}"

APPLY=0
PURGE=0
for arg in "$@"; do
    case "$arg" in
    --apply) APPLY=1 ;;
    --purge) PURGE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done
[ "$PURGE" = 1 ] && [ "$APPLY" = 0 ] && { echo "--purge requires --apply" >&2; exit 2; }

run() {
    if [ "$APPLY" = 1 ]; then echo "+ $*"; "$@"; else echo "  [dry-run] $*"; fi
}

fail() { echo "ABORT: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || fail "run as root (sudo)"
findmnt /db >/dev/null || fail "/db is not mounted -- run setup_storage_tiers.sh first"
findmnt /bulk >/dev/null || fail "/bulk is not mounted -- run setup_storage_tiers.sh first"
[ -d "$SRC/data/databases" ] || fail "$SRC/data/databases not found"

# rsync a source into a destination parent, then (under --purge) verify with a second
# pass and delete the source. --itemize-changes on the verify pass: any output line
# means the copy is incomplete, and the source is kept.
migrate() {
    local src="$1" dst_parent="$2"
    [ -e "$src" ] || { echo "  (skip, absent: $src)"; return 0; }
    run mkdir -p "$dst_parent"
    run rsync -a --info=progress2 "$src" "$dst_parent/"
    if [ "$PURGE" = 1 ] && [ "$APPLY" = 1 ]; then
        local delta
        delta="$(rsync -a --itemize-changes --dry-run "$src" "$dst_parent/" | head -5)"
        if [ -n "$delta" ]; then
            echo "  VERIFY FAILED for $src -- source kept:"
            echo "$delta"
        else
            echo "+ rm -rf $src   (verified identical on destination)"
            rm -rf "$src"
        fi
    fi
}

echo "=== 1/4 stop the serving container (store must be copied cold) ==="
run docker rm -f "$CONTAINER"

echo "=== 2/4 serving tree -> $DB_ROOT ==="
run mkdir -p "$DB_ROOT/data"
for d in databases transactions dbms cluster-state server_id; do
    migrate "$SRC/data/$d" "$DB_ROOT/data"
done
for d in conf logs plugins metrics biocypher .env; do
    migrate "$SRC/$d" "$DB_ROOT"
done

echo "=== 3/4 cold bytes -> $BULK_ROOT ==="
migrate "$SRC/biocypher-out" "$BULK_ROOT"
migrate "$SRC/data/dumps" "$BULK_ROOT"
migrate "$SRC/neo4j4-store-graveyard" "$BULK_ROOT"
migrate "$DEPRECATED_SRC" "$BULK_ROOT"
# The library mirror is COPIED, never purged: tc-lit-endpoint serves the original.
if [ -d "$LIBRARY_SRC" ]; then
    run rsync -a --info=progress2 "$LIBRARY_SRC" "$BULK_ROOT/"
fi

echo "=== 4/4 serve read-only from $DB_ROOT ==="
run docker run \
    --cpus="$SERVE_CPUS" \
    --memory="$SERVE_MEM" \
    --env=NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -d --name "$CONTAINER" \
    -p 7687:7687 -p 7474:7474 \
    --restart=unless-stopped \
    -v "$DB_ROOT/data":/var/lib/neo4j/data \
    -v "$DB_ROOT/.env":/.env \
    -v "$DB_ROOT/biocypher":/var/lib/neo4j/biocypher \
    -v "$DB_ROOT/conf":/var/lib/neo4j/conf \
    -v "$DB_ROOT/logs":/logs \
    -v "$DB_ROOT/metrics":/metrics \
    -e NEO4J_AUTH=neo4j/torchcell \
    -e NEO4J_server_databases_default__to__read__only=true \
    "$IMAGE"

if [ "$APPLY" = 1 ]; then
    echo
    echo "waiting 90s for neo4j, then verifying the served node count..."
    sleep 90
    docker exec "$CONTAINER" bash -c \
        'source /.env && cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -d torchcell "MATCH (n) RETURN count(n);"' \
        || echo "verify query failed -- check docker logs $CONTAINER"
    echo
    df -h /scratch /db /bulk
else
    echo
    echo "DRY RUN. Re-run with --apply (and optionally --purge) to execute."
fi
