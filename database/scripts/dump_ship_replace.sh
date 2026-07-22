#!/bin/bash
# database/scripts/dump_ship_replace.sh
# [[database.scripts.dump_ship_replace]]
#
# ONE command, run on GilaHyper, that replaces the served Radiant DB with the
# current GilaHyper torchcell DB: dump -> ship -> load-in-place -> verify, driven
# entirely over SSH. No second VS Code / SSH session on Radiant needed.
#
#   bash database/scripts/dump_ship_replace.sh
#
# FULL REPLACEMENT (no versioning): the Radiant torchcell DB is dropped and
# reloaded under the same name. Proven 2026-07-22 (see [[kg-build-gilahyper-state]]).
#
# Design: this is a thin orchestrator. It dumps + ships, then reuses the Radiant
# loader database/scripts/load_dump.sh -- scp'd over and run via SSH (Radiant's
# checkout is sparse and won't materialize it), so there is exactly ONE loader that
# encodes the Radiant specifics (NFS store, --user 67392, auth disabled). Do not
# duplicate the load logic here.
set -euo pipefail

# --- source (GilaHyper) ---
SRC_CONTAINER="${SRC_CONTAINER:-tc-neo4j-readonly}"   # container serving the freshly-built DB
DB="${DB:-torchcell}"
HOST_DUMP_DIR="${HOST_DUMP_DIR:-/scratch/projects/torchcell/database/data/dumps}"
CONTAINER_DUMP_DIR="/var/lib/neo4j/data/dumps"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- target (Radiant) ---
RADIANT="${RADIANT:-rocky@torchcell-database.ncsa.illinois.edu}"
RADIANT_DIR="${RADIANT_DIR:-torchcell-dumps}"          # relative to the remote HOME

echo "=== 1/4 offline dump of '$DB' on GilaHyper ($SRC_CONTAINER) ==="
docker exec "$SRC_CONTAINER" bash -c "source /.env && \
  cypher-shell -u \"\$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" -d system 'STOP DATABASE $DB;' && \
  mkdir -p $CONTAINER_DUMP_DIR && \
  neo4j-admin database dump $DB --to-path=$CONTAINER_DUMP_DIR --overwrite-destination && \
  cypher-shell -u \"\$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" -d system 'START DATABASE $DB;'"
DUMP="$HOST_DUMP_DIR/$DB.dump"
[ -f "$DUMP" ] || { echo "ERROR: dump not found at $DUMP"; exit 1; }

echo "=== 2/4 sha256 + rsync to $RADIANT:~/$RADIANT_DIR ==="
LOCAL_SHA="$(sha256sum "$DUMP" | awk '{print $1}')"
echo "  $DUMP  ($(du -h "$DUMP" | cut -f1))  sha256=$LOCAL_SHA"
ssh -o BatchMode=yes "$RADIANT" "mkdir -p ~/$RADIANT_DIR"
rsync -aP --partial "$DUMP" "$RADIANT:~/$RADIANT_DIR/"
REMOTE_SHA="$(ssh -o BatchMode=yes "$RADIANT" "sha256sum ~/$RADIANT_DIR/$DB.dump" | awk '{print $1}')"
[ "$LOCAL_SHA" = "$REMOTE_SHA" ] || { echo "✗ sha256 MISMATCH -- aborting before replace"; exit 1; }
echo "  ✓ sha256 match"

echo "=== 3/4 push the loader + run it on Radiant (full DROP + load + verify) ==="
scp -q "$SCRIPT_DIR/load_dump.sh" "$RADIANT:~/load_dump.sh"
ssh -o BatchMode=yes "$RADIANT" "DUMP=~/$RADIANT_DIR/$DB.dump DB=$DB bash ~/load_dump.sh"

echo "=== 4/4 done — Radiant '$DB' replaced with the current GilaHyper build. ==="
