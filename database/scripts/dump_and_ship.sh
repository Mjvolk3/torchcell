#!/bin/bash
# database/scripts/dump_and_ship.sh
# [[database.scripts.dump_and_ship]]
#
# GilaHyper side of the "ship the torchcell DB to Radiant" path.
# Dumps the running neo4j 5.26.28 torchcell database and rsyncs the dump to the
# Radiant VM, sha256-verified end to end. One command; safe to re-run frequently.
#
#   bash database/scripts/dump_and_ship.sh
#
# Proven 2026-07-21 (see [[kg-build-gilahyper-state]] + the Radiant handoff note).
# Pair: database/scripts/load_dump.sh runs on Radiant to load what this ships.
#
# WHY the details matter (all learned the hard way this session):
#  - neo4j 5.x syntax is `neo4j-admin database dump <db> --to-path=<dir>` (NOT the 4.4
#    `dump --database= --to=`). The 4.4-era database/scripts/export_database.sh is stale.
#  - dump to a HOST-MOUNTED dir (container /var/lib/neo4j/data -> host database/data) so
#    the .dump lands on disk we can rsync, without a slow `docker cp`.
#  - the dump is already compressed -> rsync WITHOUT -z (wastes CPU for no gain).
#  - sha256 both ends (provenance rule): a silently-truncated transfer must never load.
set -euo pipefail

CONTAINER="${CONTAINER:-tc-neo4j}"
DB="${DB:-torchcell}"
# container /var/lib/neo4j/data is bind-mounted from the host build tree below:
HOST_DUMP_DIR="${HOST_DUMP_DIR:-/scratch/projects/torchcell/database/data/dumps}"
CONTAINER_DUMP_DIR="/var/lib/neo4j/data/dumps"
RADIANT="${RADIANT:-rocky@torchcell-database.ncsa.illinois.edu}"
RADIANT_DIR="${RADIANT_DIR:-~/torchcell-dumps}"

echo "=== 1/4 offline dump of '$DB' (stop -> dump -> start) ==="
docker exec "$CONTAINER" bash -c "source /.env && \
  cypher-shell -u \"\$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" -d system 'STOP DATABASE $DB;' && \
  mkdir -p $CONTAINER_DUMP_DIR && \
  neo4j-admin database dump $DB --to-path=$CONTAINER_DUMP_DIR --overwrite-destination && \
  cypher-shell -u \"\$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" -d system 'START DATABASE $DB;'"

DUMP="$HOST_DUMP_DIR/$DB.dump"
[ -f "$DUMP" ] || { echo "ERROR: dump not found on host at $DUMP"; exit 1; }

echo "=== 2/4 sha256 anchor ==="
LOCAL_SHA="$(sha256sum "$DUMP" | awk '{print $1}')"
echo "  $DUMP  ($(du -h "$DUMP" | cut -f1))"
echo "  local sha256: $LOCAL_SHA"

echo "=== 3/4 rsync to $RADIANT:$RADIANT_DIR (resumable) ==="
ssh -o BatchMode=yes "$RADIANT" "mkdir -p $RADIANT_DIR"
rsync -aP --partial "$DUMP" "$RADIANT:$RADIANT_DIR/"

echo "=== 4/4 remote sha256 verify ==="
REMOTE_SHA="$(ssh -o BatchMode=yes "$RADIANT" "sha256sum $RADIANT_DIR/$DB.dump" | awk '{print $1}')"
echo "  remote sha256: $REMOTE_SHA"
if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
    echo "✓ sha256 MATCH — $DB.dump shipped to $RADIANT:$RADIANT_DIR"
    echo "  next: on Radiant run  bash database/scripts/load_dump.sh"
else
    echo "✗ sha256 MISMATCH — transfer corrupt; re-run (rsync --partial resumes)"
    exit 1
fi
