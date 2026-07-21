#!/bin/bash
# database/scripts/load_dump.sh
# [[database.scripts.load_dump]]
#
# Radiant side of the "ship the torchcell DB to Radiant" path.
# Loads a shipped dump into the Radiant neo4j 5.26.28 container. One command; safe
# to re-run frequently (each run cleanly replaces the previous torchcell store).
#
#   bash database/scripts/load_dump.sh            # uses ~/torchcell-dumps/torchcell.dump
#   DUMP=/path/to/x.dump bash database/scripts/load_dump.sh
#
# Assumes the 5.26.28 container is already running (image pulled from Docker Hub,
# conf migrated, TLS in place -- see the Radiant handoff note). Pair: dump_and_ship.sh
# ran on GilaHyper to produce + rsync the dump here.
#
# WHY the details matter (proven 2026-07-21, see [[kg-build-gilahyper-state]]):
#  - neo4j 5.x syntax is `neo4j-admin database load <db> --from-path=<dir>` (NOT the 4.4
#    `load --database= --from=`). The 4.4-era import_database_openstack.sh is stale.
#  - GOTCHA 1 "The database is in use": a leftover store dir from a killed/prior load
#    blocks the importer even with --overwrite-destination. FIX: rm the store dir first.
#  - GOTCHA 2 DIRTY / AccessDenied: `docker exec` runs as ROOT, so a root-run load writes
#    root-owned store files that the neo4j server (uid 7474) cannot start OR drop ->
#    database goes DIRTY. FIX: chown -R neo4j:neo4j the store BEFORE `CREATE DATABASE`.
#  - admin cypher (DROP/CREATE/SHOW DATABASES) MUST target -d system, not the db context.
set -uo pipefail

CONTAINER="${CONTAINER:-tc-neo4j}"
DB="${DB:-torchcell}"
DUMP="${DUMP:-$HOME/torchcell-dumps/$DB.dump}"        # dump on the Radiant HOST
CONTAINER_DUMP_DIR="/var/lib/neo4j/data/dumps"        # host-mounted; where we stage it

[ -f "$DUMP" ] || { echo "ERROR: dump not found at $DUMP"; exit 1; }
docker ps -q -f name="$CONTAINER" >/dev/null || { echo "ERROR: container '$CONTAINER' not running"; exit 1; }
cyp() { docker exec "$CONTAINER" bash -c "source /.env && cypher-shell -u \"\$NEO4J_USER\" -p \"\$NEO4J_PASSWORD\" $*"; }

echo "=== 1/6 stage the dump into the container ($(du -h "$DUMP" | cut -f1)) ==="
docker exec "$CONTAINER" mkdir -p "$CONTAINER_DUMP_DIR"
docker cp "$DUMP" "$CONTAINER:$CONTAINER_DUMP_DIR/$DB.dump"

echo "=== 2/6 drop any prior '$DB' + remove its store (the 'in use' fix) ==="
cyp "-d system 'STOP DATABASE $DB;'" 2>/dev/null || true
# rm the store dirs as root (a root-run load may have left root-owned files a DROP can't delete)
docker exec "$CONTAINER" bash -c "rm -rf /var/lib/neo4j/data/databases/$DB /var/lib/neo4j/data/transactions/$DB"
cyp "-d system 'DROP DATABASE $DB IF EXISTS;'" 2>/dev/null || true

echo "=== 3/6 load (neo4j 5.x) ==="
docker exec "$CONTAINER" neo4j-admin database load "$DB" \
    --from-path="$CONTAINER_DUMP_DIR" --overwrite-destination

echo "=== 4/6 chown store to neo4j (the DIRTY fix; docker exec ran load as root) ==="
docker exec "$CONTAINER" bash -c "chown -R neo4j:neo4j /var/lib/neo4j/data/databases/$DB /var/lib/neo4j/data/transactions/$DB"

echo "=== 5/6 register + wait ==="
cyp "-d system 'CREATE DATABASE $DB;'"
sleep 10
cyp "-d system \"SHOW DATABASE $DB YIELD name, currentStatus, statusMessage;\""

echo "=== 6/6 verify counts (expect the GilaHyper source counts) ==="
cyp "-d $DB 'MATCH (n) RETURN count(n) AS nodes;'"
cyp "-d $DB 'MATCH ()-[r]->() RETURN count(r) AS rels;'"

echo "✓ load complete. If serving read-only, follow with the ALTER DATABASE ... SET ACCESS READ ONLY"
echo "  + create_readonly_users.sh steps (see database/scripts/import_database_openstack.sh)."
