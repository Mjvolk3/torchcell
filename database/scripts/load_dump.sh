#!/bin/bash
# database/scripts/load_dump.sh
# [[database.scripts.load_dump]]
#
# Radiant side of the "ship the torchcell DB to Radiant" path.
# Loads a shipped dump into the Radiant neo4j 5.26.28 deployment. One command;
# safe to re-run (each run cleanly replaces the previous torchcell store).
#
#   bash database/scripts/load_dump.sh            # dump: ~/torchcell-dumps/torchcell.dump
#   DUMP=/path/to/x.dump bash database/scripts/load_dump.sh
#
# Pair: dump_and_ship.sh runs on GilaHyper to produce + rsync the dump here.
#
# ARCHITECTURE (proven 2026-07-22 — supersedes the earlier local-store version):
#  - The store is ~46 GB and CANNOT fit the 40 GB local disk. It lives on the
#    Taiga NFS mount at $STORE_HOST (= /mnt/delta_bbub/.../torchcell_openstack/neo4j-data),
#    which is Delta /projects/bbub. Only the docker image stays on local disk.
#  - The mount's NFSv4 ACL only lets ROOT and the OWNER uid 67392 (mjvolk3, gid 202)
#    write — NOT uid 7474, even with ACL grants. So the neo4j container runs as
#    --user 67392:202 (the entrypoint runs neo4j as that uid instead of gosu-dropping
#    to 7474), and the store dir is owned 67392:202. Loading AS 67392 means store
#    files are owner-writable from birth — no chown-DIRTY gotcha.
#  - AUTH IS DISABLED (dbms.security.auth_enabled=false) -> cypher-shell takes NO -u/-p.
#  - 5.x syntax: `neo4j-admin database load <db> --from-path=<dir> --overwrite-destination`.
#  - admin cypher (STOP/DROP/CREATE/SHOW DATABASE) MUST target -d system.
set -uo pipefail

CONTAINER="${CONTAINER:-tc-neo4j}"
DB="${DB:-torchcell}"
IMAGE="${IMAGE:-michaelvolk/tc-neo4j:5.26.28}"
DUMP="${DUMP:-$HOME/torchcell-dumps/$DB.dump}"                                   # dump on the Radiant HOST
STORE_HOST="${STORE_HOST:-/mnt/delta_bbub/mjvolk3/torchcell_openstack/neo4j-data}"  # store on the mount
RUN_AS="${RUN_AS:-67392:202}"                                                    # mjvolk3 (mount owner)

[ -f "$DUMP" ] || { echo "ERROR: dump not found at $DUMP"; exit 1; }
docker ps -q -f name="$CONTAINER" >/dev/null || { echo "ERROR: container '$CONTAINER' not running"; exit 1; }
DUMP_DIR="$(dirname "$DUMP")"
# cypher-shell WITHOUT credentials (auth disabled); admin ops target system
cyp() { docker exec "$CONTAINER" cypher-shell -d system "$@"; }

echo "=== 1/6 drop any prior '$DB' so its store can be replaced ==="
cyp "STOP DATABASE $DB;" 2>/dev/null || true
cyp "DROP DATABASE $DB IF EXISTS;" 2>/dev/null || true

echo "=== 2/6 remove the old '$DB' store dir on the mount (as 67392) ==="
docker run --rm --user "$RUN_AS" -v "$STORE_HOST:/data" --entrypoint bash "$IMAGE" \
    -c "rm -rf /data/databases/$DB /data/transactions/$DB"

echo "=== 3/6 load the dump into the mount-store as 67392 ($(du -h "$DUMP" | cut -f1)) ==="
docker run --rm --user "$RUN_AS" \
    -v "$STORE_HOST:/data" -v "$DUMP_DIR:/dumps:ro" \
    --entrypoint neo4j-admin "$IMAGE" \
    database load "$DB" --from-path=/dumps --overwrite-destination

echo "=== 4/6 register '$DB' (adopts the loaded store) ==="
cyp "CREATE DATABASE $DB IF NOT EXISTS;"

echo "=== 5/6 wait for '$DB' online ==="
for _ in $(seq 1 30); do
    st="$(cyp "SHOW DATABASE $DB YIELD currentStatus;" 2>/dev/null | tail -1)"
    echo "  $st"
    echo "$st" | grep -q online && break
    sleep 5
done

echo "=== 6/6 verify counts (expect the GilaHyper source counts) ==="
docker exec "$CONTAINER" cypher-shell -d "$DB" "MATCH (n) RETURN count(n) AS nodes;"
docker exec "$CONTAINER" cypher-shell -d "$DB" "MATCH ()-[r]->() RETURN count(r) AS rels;"
echo "✓ load complete (nodes should be 7,158,410 / rels 20,996,966 for the 2026-07 build)."
