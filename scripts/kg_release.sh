#!/usr/bin/env bash
# scripts/kg_release.sh
# [[scripts.kg_release]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/kg_release.sh
#
# Knowledge-graph release artifacts: an online backup of the served store, archived to
# /bulk with its manifest and sha256, shipped to Taiga, listed, and purged on a
# retention window that spares tagged releases.
#
#   kg_release.sh backup            online backup of the served database straight into
#                                   $ARCHIVE_ROOT/<release>/, which the serving container
#                                   mounts at /kg-releases (no downtime: Enterprise
#                                   `neo4j-admin database backup`; the work directory
#                                   holds the whole store before compressing, so this
#                                   must be the big /bulk RAID, never /db)
#   kg_release.sh archive           finish the release directory: kg_manifest.json +
#                                   release.json + SHA256SUMS beside the .backup
#   kg_release.sh ship [<release>]  rsync one archived release to Taiga ($TAIGA_DEST)
#   kg_release.sh list              releases on /bulk and on Taiga, with KEEP tags
#   kg_release.sh keep <release> "<why>"   tag a release so purge never removes it
#   kg_release.sh purge [--dry-run] remove UNTAGGED releases older than $KEEP_DAYS from
#                                   /bulk and Taiga; dry run prints what it would do
#   kg_release.sh deploy <release> [--pin]
#                                   serve an archived release from THIS host's DBMS:
#                                   pull it from Taiga when $ARCHIVE_ROOT lacks it,
#                                   verify SHA256SUMS, `neo4j-admin database restore`
#                                   it into database kg-<release> beside what is served,
#                                   create it, check its KgRelease node against
#                                   release.json, retarget the `latest` alias (and
#                                   `pinned` with --pin), then `ops.sh sync`. This is
#                                   how the served version follows a release on a host
#                                   that did not build it (Radiant, once its store sits
#                                   on a block volume; Neo4j does not run on NFS).
#   kg_release.sh restore-test <release>
#                                   the deploy's restore and checks into a throwaway
#                                   database kgtest-<release>, which is dropped at the
#                                   end; aliases untouched. Proves the artifact restores
#                                   before it is trusted as the sole copy.
#   kg_release.sh retag <release> <tag>
#                                   pair a release built from an untagged commit with
#                                   the package tag cut afterwards: `releases retag`
#                                   rewrites the committed snapshot and the manifest
#                                   (refused unless the surface at the tag reproduces
#                                   every served closure), then the KgRelease node is
#                                   rewritten from the manifest. Commit the snapshot
#                                   as `DB(kg): ...` and regenerate the compat page.
#
# deploy / restore-test / retag touch the DBMS, so on GilaHyper they run under slurm:
#   sbatch -J kg-deploy -c 4 --mem=16G --time=4:00:00 --wrap "bash scripts/kg_release.sh deploy <release>"
#
# Layout of one release directory:
#   <release>/torchcell-<ts>.backup   the backup artifact (compressed; `neo4j-admin
#                                     database restore --from-path` loads it)
#   <release>/kg_manifest.json        the served manifest at archive time
#   <release>/release.json            the KgRelease node's properties
#   <release>/SHA256SUMS              sha256 of every file above
#   <release>/KEEP                    "<why>", written by `keep`; purge skips the dir
#
# The release id comes from the served store's KgRelease node, so the artifact names
# what it holds. The backup client runs INSIDE the serving container (its backup port
# listens on localhost only) as the neo4j user, writing to the /kg-releases mount
# (= $ARCHIVE_ROOT on the host; mounted outside NEO4J_HOME so the entrypoint's
# chmod 700 sweep never hides it from the host).
#
# Env: NEO4J_CONTAINER (tc-neo4j-readonly), NEO4J_DATABASE (torchcell),
#      ARCHIVE_ROOT (/bulk/kg-releases),
#      TAIGA_HOST (rocky@141.142.216.218),
#      TAIGA_DEST (/mnt/zhao5/mjvolk3/projects/torchcell/kg-releases),
#      KEEP_DAYS (60), KG_MANIFEST (/scratch/projects/torchcell/database/kg_manifest.json),
#      IMAGE (michaelvolk/tc-neo4j:5.26.28-browser.1) for the root helper container.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${OPS_PYTHON:-$HOME/miniconda3/envs/torchcell/bin/python}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-tc-neo4j-readonly}"
NEO4J_DATABASE="${NEO4J_DATABASE:-torchcell}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/bulk/kg-releases}"
STAGING_CONT="/kg-releases"  # $ARCHIVE_ROOT as the serving container mounts it
TAIGA_HOST="${TAIGA_HOST:-rocky@141.142.216.218}"
TAIGA_DEST="${TAIGA_DEST:-/mnt/zhao5/mjvolk3/projects/torchcell/kg-releases}"
KEEP_DAYS="${KEEP_DAYS:-60}"
KG_MANIFEST="${KG_MANIFEST:-/scratch/projects/torchcell/database/kg_manifest.json}"
IMAGE="${IMAGE:-michaelvolk/tc-neo4j:5.26.28-browser.1}"
CS="cypher-shell -u ${NEO4J_USER:-neo4j} -p ${NEO4J_PASSWORD:-torchcell}"

served_release() {
    docker exec "$NEO4J_CONTAINER" $CS -d "$NEO4J_DATABASE" --format plain \
        "MATCH (r:KgRelease) RETURN r.release;" | tail -1 | tr -d '"'
}

release_json() {  # the served database's KgRelease node as JSON (status --json is {host: [databases]})
    PYTHONWARNINGS=ignore "$PY" -m torchcell.knowledge_graphs.releases status --json 2>/dev/null \
        | "$PY" -c "import json,sys; hosts=json.load(sys.stdin); dbs=[d for v in hosts.values() for d in v]; [print(json.dumps(d['release'], indent=1)) for d in dbs if d['name']=='$NEO4J_DATABASE' and d['release']]"
}

cmd_backup() {
    local release
    release=$(served_release)
    [[ -n "$release" && "$release" != "null" ]] || { echo "the served store carries no KgRelease node; write it first"; exit 1; }
    docker inspect "$NEO4J_CONTAINER" --format '{{range .Mounts}}{{.Destination}}{{"\n"}}{{end}}' | grep -qx "$STAGING_CONT" \
        || { echo "$NEO4J_CONTAINER does not mount $STAGING_CONT; relaunch it with -v $ARCHIVE_ROOT:$STAGING_CONT"; exit 1; }
    echo "release $release -> $ARCHIVE_ROOT/$release (online backup, no downtime; runs detached, watch backup.log)"
    docker exec -u neo4j -d "$NEO4J_CONTAINER" bash -c "mkdir -p $STAGING_CONT/$release && date > $STAGING_CONT/$release/backup.log \
        && neo4j-admin database backup --to-path=$STAGING_CONT/$release --from=localhost:6362 $NEO4J_DATABASE >> $STAGING_CONT/$release/backup.log 2>&1; \
        echo \"exit \$?\" >> $STAGING_CONT/$release/backup.log; date >> $STAGING_CONT/$release/backup.log"
    sleep 5
    docker exec "$NEO4J_CONTAINER" bash -c "tail -3 $STAGING_CONT/$release/backup.log; ls -la $STAGING_CONT/$release"
}

cmd_archive() {
    local release dest
    release=$(served_release)
    dest="$ARCHIVE_ROOT/$release"
    grep -q '^exit 0' "$dest/backup.log" 2>/dev/null \
        || { echo "no finished backup under $dest (backup.log lacks 'exit 0')"; exit 1; }
    echo "finishing release directory $dest"
    release_json > "/tmp/release-$release.json"
    # the tree is uid-7474 owned; a root helper container writes beside the artifact
    docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive -v "$KG_MANIFEST":/kg_manifest.json:ro \
        -v "/tmp/release-$release.json":/release.json:ro "$IMAGE" -c "set -e
        cd /archive/$release
        cp /kg_manifest.json kg_manifest.json
        cp /release.json release.json
        sha256sum *.backup kg_manifest.json release.json > SHA256SUMS
        chmod -R a+rX /archive/$release
        ls -la /archive/$release"
    echo "archived $dest"
}

cmd_ship() {
    local release="${1:-$(served_release)}"
    [[ -d "$ARCHIVE_ROOT/$release" ]] || { echo "no archived release at $ARCHIVE_ROOT/$release"; exit 1; }
    echo "shipping $release -> $TAIGA_HOST:$TAIGA_DEST/$release"
    ssh "$TAIGA_HOST" "mkdir -p $TAIGA_DEST/$release"
    # -rlpt, not -a: Taiga's setgid group directories refuse chgrp, and -a's owner/group
    # preservation would turn that refusal into rsync exit 23 after a complete copy
    rsync -rlpt --partial --info=progress2 "$ARCHIVE_ROOT/$release/" "$TAIGA_HOST:$TAIGA_DEST/$release/"
    ssh "$TAIGA_HOST" "cd $TAIGA_DEST/$release && sha256sum -c SHA256SUMS"
    echo "shipped and verified $release"
}

cmd_keep() {
    local release="$1" why="$2"
    [[ -d "$ARCHIVE_ROOT/$release" ]] || { echo "no archived release at $ARCHIVE_ROOT/$release"; exit 1; }
    docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c "printf '%s\n' \"$why\" > /archive/$release/KEEP"
    ssh "$TAIGA_HOST" "[ -d $TAIGA_DEST/$release ] && printf '%s\n' \"$why\" > $TAIGA_DEST/$release/KEEP || true"
    echo "kept $release: $why"
}

_list_tree() {  # <label> <ls output of the root: name per line> <root>
    local label="$1" root="$2"
    shift 2
    for name in "$@"; do
        local keep="" size=""
        keep=$(cat "$root/$name/KEEP" 2>/dev/null || true)
        size=$(du -sh "$root/$name" 2>/dev/null | cut -f1)
        printf '  %-10s %-22s %-7s %s\n' "$label" "$name" "${size:-?}" "${keep:+KEEP: $keep}"
    done
}

cmd_list() {
    echo "== releases on $ARCHIVE_ROOT =="
    if [[ -d "$ARCHIVE_ROOT" ]]; then
        docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c \
            'cd /archive && for d in */; do d=${d%/}; printf "  %-10s %-22s %-7s %s\n" bulk "$d" "$(du -sh "$d" | cut -f1)" "$(cat "$d/KEEP" 2>/dev/null | sed "s/^/KEEP: /")"; done'
    fi
    echo "== releases on $TAIGA_HOST:$TAIGA_DEST =="
    ssh "$TAIGA_HOST" "cd $TAIGA_DEST 2>/dev/null && for d in */; do d=\${d%/}; printf '  %-10s %-22s %-7s %s\n' taiga \"\$d\" \"\$(du -sh \"\$d\" | cut -f1)\" \"\$(cat \"\$d/KEEP\" 2>/dev/null | sed 's/^/KEEP: /')\"; done" || echo "  (none)"
}

# Age is read from the release id's date prefix (YYYY.MM.DD), the build date.
_expired() {  # <release> -> 0 when older than KEEP_DAYS
    local date="${1%%-*}" epoch cutoff
    epoch=$(date -d "${date//./-}" +%s 2>/dev/null) || return 1
    cutoff=$(( $(date +%s) - KEEP_DAYS * 86400 ))
    (( epoch < cutoff ))
}

cmd_purge() {
    local dry=0
    [[ "${1:-}" == "--dry-run" ]] && dry=1
    local names
    names=$(docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c 'cd /archive && ls -d */ 2>/dev/null | tr -d /')
    for name in $names; do
        if docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c "[ -f /archive/$name/KEEP ]"; then
            echo "keep   $name (tagged)"; continue
        fi
        if ! _expired "$name"; then
            echo "keep   $name (younger than $KEEP_DAYS days)"; continue
        fi
        if (( dry )); then
            echo "WOULD purge $name from $ARCHIVE_ROOT and $TAIGA_DEST"
        else
            docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c "rm -rf /archive/$name"
            ssh "$TAIGA_HOST" "[ -f $TAIGA_DEST/$name/KEEP ] || rm -rf $TAIGA_DEST/$name"
            echo "purged $name"
        fi
    done
}

_db_name() {  # <prefix> <release> -> a Neo4j database name: letters, digits, dashes
    printf '%s-%s' "$1" "${2//./-}"
}

_cypher() {  # <database> <statement...>
    local db="$1"; shift
    docker exec "$NEO4J_CONTAINER" $CS -d "$db" --format plain "$*"
}

_release_dir_ready() {  # <release> -> pull from Taiga when absent, then verify checksums
    local release="$1" dest="$ARCHIVE_ROOT/$release"
    if ! docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c "[ -f /archive/$release/SHA256SUMS ]"; then
        echo "no archived $release under $ARCHIVE_ROOT; pulling from $TAIGA_HOST:$TAIGA_DEST/$release"
        ssh "$TAIGA_HOST" "[ -f $TAIGA_DEST/$release/SHA256SUMS ]" || { echo "Taiga has no archived $release either"; exit 1; }
        docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c "mkdir -p /archive/$release && chmod a+rwx /archive/$release"
        rsync -rlpt --partial --info=progress2 "$TAIGA_HOST:$TAIGA_DEST/$release/" "$dest/"
    fi
    docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c "cd /archive/$release && sha256sum -c SHA256SUMS"
}

# Restore <release>'s artifact into database <db> of the serving DBMS and check it.
# The DBMS stays online: `neo4j-admin database restore` writes a database that does not
# exist yet, `CREATE DATABASE` brings it up, and the new database inherits
# server.databases.default_to_read_only. The checks compare the restored KgRelease node
# with release.json (release id, version, dataset count) so the artifact is proven to be
# what its directory says it is.
_restore_into() {  # <release> <db>
    local release="$1" db="$2" artifact free_gb need_gb
    artifact=$(docker exec "$NEO4J_CONTAINER" bash -c "ls $STAGING_CONT/$release/*.backup" | head -1)
    [[ -n "$artifact" ]] || { echo "no .backup artifact under $STAGING_CONT/$release"; exit 1; }
    if _cypher system "SHOW DATABASES YIELD name RETURN name;" | grep -qx "\"$db\""; then
        echo "database $db already exists in $NEO4J_CONTAINER; drop it first or deploy another release"; exit 1
    fi
    free_gb=$(docker exec "$NEO4J_CONTAINER" df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    need_gb="${MIN_FREE_GB:-300}"
    [ "$free_gb" -ge "$need_gb" ] || { echo "/data has ${free_gb}G free, need ${need_gb}G for a restored store"; exit 1; }
    echo "restoring $artifact -> database $db ($(date))"
    docker exec -u neo4j "$NEO4J_CONTAINER" neo4j-admin database restore --from-path="$artifact" "$db"
    _cypher system "CREATE DATABASE \`$db\` WAIT;"
    local got want_release want_version want_n got_n
    got=$(_cypher "$db" "MATCH (r:KgRelease) RETURN r.release, r.version, r.n_datasets;" | tail -1 | tr -d '" ')
    want_release=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['release'])" "$ARCHIVE_ROOT/$release/release.json")
    want_version=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "$ARCHIVE_ROOT/$release/release.json")
    # release.json is the KgRelease model (datasets as a map), not the node's flat properties
    want_n=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['datasets']))" "$ARCHIVE_ROOT/$release/release.json")
    got_n=$(_cypher "$db" "MATCH (d:Dataset) RETURN count(d);" | tail -1)
    [[ "$got" == "$want_release,$want_version,$want_n" ]] \
        || { echo "restored $db carries KgRelease ($got), release.json says $want_release,$want_version,$want_n"; exit 1; }
    [[ "$got_n" == "$want_n" ]] || { echo "restored $db has $got_n Dataset nodes, release.json says $want_n"; exit 1; }
    echo "restored $db: release $want_release (KG $want_version), $got_n datasets ($(date))"
}

cmd_deploy() {
    local release="${1:-}" pin=0
    [[ -n "$release" ]] || { echo "usage: $0 deploy <release> [--pin]"; exit 2; }
    [[ "${2:-}" == "--pin" ]] && pin=1
    local db; db=$(_db_name kg "$release")
    _release_dir_ready "$release"
    _restore_into "$release" "$db"
    _cypher system "ALTER ALIAS latest SET DATABASE \`$db\`;"
    (( pin )) && _cypher system "ALTER ALIAS pinned SET DATABASE \`$db\`;"
    echo "latest -> $db${pin:+ (pinned too)}; the previous database stays for rollback (ALTER ALIAS latest SET DATABASE <name>)"
    PYTHONWARNINGS=ignore "$PY" -m torchcell.knowledge_graphs.releases --repo "$REPO_ROOT" status --label "$(hostname -s)" 2>/dev/null | grep -vE '^INFO --|DeprecationWarning|^<frozen|^$'
    bash "$REPO_ROOT/scripts/ops.sh" sync
}

cmd_restore_test() {
    local release="${1:-}"
    [[ -n "$release" ]] || { echo "usage: $0 restore-test <release>"; exit 2; }
    local db; db=$(_db_name kgtest "$release")
    _release_dir_ready "$release"
    _restore_into "$release" "$db"
    # The throwaway database carries the same KgRelease node as the served one, so a
    # client naming the release id would see two candidates while it exists; drop it.
    _cypher system "DROP DATABASE \`$db\` DESTROY DATA WAIT;"
    echo "restore test passed for $release; $db dropped"
}

cmd_retag() {
    local release="${1:-}" tag="${2:-}"
    [[ -n "$release" && -n "$tag" ]] || { echo "usage: $0 retag <release> <tag>"; exit 2; }
    [[ "$(served_release)" == "$release" ]] || { echo "the served store is $(served_release), not $release; retag only the served release here"; exit 1; }
    PYTHONPATH="$REPO_ROOT" "$PY" -m torchcell.knowledge_graphs.releases retag --release "$release" --tag "$tag" \
        --repo-root "$REPO_ROOT" --manifest "$KG_MANIFEST"
    local built_at n_nodes
    built_at=$(_cypher "$NEO4J_DATABASE" "MATCH (r:KgRelease) RETURN r.built_at;" | tail -1 | tr -d '"')
    n_nodes=$(_cypher "$NEO4J_DATABASE" "MATCH (r:KgRelease) RETURN r.n_nodes;" | tail -1)
    _cypher system "CALL dbms.setConfigValue('server.databases.writable', '$NEO4J_DATABASE');"
    PYTHONPATH="$REPO_ROOT" "$PY" -m torchcell.knowledge_graphs.releases write-node --manifest "$KG_MANIFEST" \
        --database "$NEO4J_DATABASE" --built-at "$built_at" --n-nodes "$n_nodes"
    _cypher system "CALL dbms.setConfigValue('server.databases.writable', '');"
    _cypher "$NEO4J_DATABASE" "MATCH (r:KgRelease) RETURN r.release, r.torchcell_version, r.torchcell_tag;" | tail -1
    echo "retagged $release -> $tag; commit database/releases/$release.json as 'DB(kg): ...' and run python scripts/kg_compat_page.py"
}

case "${1:-}" in
    backup)  cmd_backup ;;
    archive) cmd_archive ;;
    ship)    shift; cmd_ship "$@" ;;
    keep)    shift; cmd_keep "$@" ;;
    list)    cmd_list ;;
    purge)   shift; cmd_purge "$@" ;;
    deploy)  shift; cmd_deploy "$@" ;;
    restore-test) shift; cmd_restore_test "$@" ;;
    retag)   shift; cmd_retag "$@" ;;
    *) sed -n '2,62p' "$0" >&2; exit 2 ;;
esac
