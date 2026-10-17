#!/usr/bin/env bash
# scripts/kg_release.sh
# [[scripts.kg_release]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/kg_release.sh
#
# Knowledge-graph release artifacts: an online backup of the served store, archived to
# /bulk with its manifest and sha256, shipped to Taiga, listed, and purged on a
# retention window that spares tagged releases; and serving an archived release, on
# GilaHyper or on Radiant straight from the Taiga mount the archive sits on.
#
#   kg_release.sh backup            online backup of the served database straight into
#                                   $ARCHIVE_ROOT/<release>/, which the serving container
#                                   mounts at /kg-releases (no downtime: Enterprise
#                                   `neo4j-admin database backup`; the work directory
#                                   holds the whole store before compressing, so this
#                                   must be the big /bulk RAID, never /db)
#   kg_release.sh archive           finish the release directory: kg_manifest.json +
#                                   release.json + SHA256SUMS beside the .backup
#   kg_release.sh ship [<release>]  rsync one archived release to Taiga ($TAIGA_DEST,
#                                   written through the Radiant VM's NFS mount, so it
#                                   lands directly where deploy --from-archive-mount
#                                   restores from) and verify SHA256SUMS there
#   kg_release.sh list              releases on /bulk and on Taiga, with KEEP tags
#   kg_release.sh keep <release> "<why>"   tag a release so purge never removes it
#   kg_release.sh purge [--dry-run] remove UNTAGGED releases older than $KEEP_DAYS from
#                                   /bulk and Taiga; dry run prints what it would do
#   kg_release.sh deploy <release> [--pin] [--dry-run]
#                                   serve an archived release from THIS host's DBMS:
#                                   pull it from Taiga when $ARCHIVE_ROOT lacks it,
#                                   verify SHA256SUMS, `neo4j-admin database restore`
#                                   it into database kg-<release> beside what is served,
#                                   create it, check it against release.json (release
#                                   id, version, every dataset's experiment count, the
#                                   node total), retarget the `latest` alias (and
#                                   `pinned` with --pin), then `ops.sh sync`.
#                                   --dry-run checks the archive, the name and the free
#                                   space and restores nothing.
#   kg_release.sh deploy <release> --from-archive-mount [--pin] [--dry-run]
#                                   the same for the Radiant DBMS ($SERVE_HOST), with
#                                   no copy of the release anywhere first: SHA256SUMS is
#                                   verified where `ship` put it ($TAIGA_DEST), and a
#                                   detached helper container restores the artifact in
#                                   place (archive mounted read-only at /kg-releases)
#                                   into the DBMS data directory on the same Taiga mount
#                                   ($SERVE_DATA), unpacking into a temp directory there
#                                   too. When the serving container already runs on
#                                   $SERVE_DATA, the database is created, checked, given
#                                   the committed snapshot's package pairing, set READ
#                                   ONLY, and `latest`, `torchcell` and the default
#                                   database move to it (`pinned` too with --pin). When
#                                   it does not yet (the first move onto the mount), the
#                                   run ends after the restore and serve-swap follows.
#   kg_release.sh serve-swap --latest <release> [--pinned <release>] [--dry-run]
#                                   move Radiant serving onto $SERVE_DATA, once: a
#                                   staging DBMS on $SERVE_DATA (no published ports, 2G
#                                   memory, a copy of the serving conf) registers the
#                                   restored releases (create, check, pairing, READ
#                                   ONLY), the aliases and the default database; it is
#                                   removed, the serving container is stopped and kept as
#                                   <name>-pre-<timestamp> (restart=no; its store is the
#                                   rollback copy), and <name> is relaunched with the
#                                   same image, user, ports, env, conf, certificates and
#                                   plugins on $SERVE_DATA / $SERVE_LOGS with the archive
#                                   read-only at /kg-releases. Downtime: one stop, one
#                                   start. `pinned` follows --pinned, else --latest.
#   kg_release.sh serve-point --latest <release> [--pinned <release>] [--dry-run]
#                                   point `latest`, `torchcell` and the default database
#                                   (and `pinned`) of the archive-mount DBMS at releases
#                                   it already serves: the alias-level rollback
#   kg_release.sh serve-rollback <kept container> [--dry-run]
#                                   undo serve-swap: stop the archive-mount container,
#                                   keep it as <name>-archive-mount-<timestamp>
#                                   (restart=no), rename the kept pre-swap container back
#                                   to <name> and start it on its old store
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
# The --from-archive-mount, serve-swap, serve-point and serve-rollback commands run on
# GilaHyper too (the torchcell environment, the committed snapshots and `ops.sh sync`
# live there) and drive $SERVE_HOST over ssh; the restore reads and writes only on the
# Taiga mount, so nothing large crosses the ssh connection, and the restore container
# runs detached, so a dropped connection does not stop it.
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
#      IMAGE (michaelvolk/tc-neo4j:5.26.28-browser.1) for the root helper container,
#      MIN_FREE_GB (300 on this host's /data; 1000 on the Taiga mount),
#      archive-mount serving: SERVE_HOST ($TAIGA_HOST; `local` runs on this host),
#      SERVE_CONTAINER (tc-neo4j),
#      SERVE_DATA (/mnt/zhao5/mjvolk3/projects/torchcell/neo4j-data),
#      SERVE_LOGS (/mnt/zhao5/mjvolk3/projects/torchcell/neo4j-logs),
#      SERVE_BOLT (neo4j+ssc://localhost:7687; Radiant's bolt requires TLS),
#      SERVE_RESTART (unless-stopped, the policy serve-rollback restores),
#      SNAPSHOT_ROOT (this checkout; holds database/releases/<release>.json).
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
SERVE_HOST="${SERVE_HOST:-$TAIGA_HOST}"
SERVE_CONTAINER="${SERVE_CONTAINER:-tc-neo4j}"
SERVE_DATA="${SERVE_DATA:-/mnt/zhao5/mjvolk3/projects/torchcell/neo4j-data}"
SERVE_LOGS="${SERVE_LOGS:-/mnt/zhao5/mjvolk3/projects/torchcell/neo4j-logs}"
SERVE_BOLT="${SERVE_BOLT:-neo4j+ssc://localhost:7687}"
SERVE_RESTART="${SERVE_RESTART:-unless-stopped}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$REPO_ROOT}"
RELEASE_RE='^[0-9]{4}\.[0-9]{2}\.[0-9]{2}-[0-9a-f]{7,}$'

WORK=""        # per-run scratch for query output and release.json copies (_work)
ON_FAIL=""     # what to tell the operator when a run stops partway
_on_exit() {
    local rc=$?
    if [[ -n "$WORK" ]]; then rm -rf "$WORK"; fi
    if (( rc != 0 )) && [[ -n "$ON_FAIL" ]]; then printf '\n%s\n' "$ON_FAIL" >&2; fi
}
trap _on_exit EXIT

die() { echo "$*" >&2; exit 1; }

_work() { [[ -n "$WORK" ]] || WORK=$(mktemp -d /tmp/kg-release.XXXXXX); }

_releases() {  # this checkout's releases CLI (a worktree's code, not the installed package's)
    PYTHONPATH="$REPO_ROOT" PYTHONWARNINGS=ignore "$PY" -m torchcell.knowledge_graphs.releases "$@"
}

_release_id() {  # <release>: refuse anything that is not <YYYY.MM.DD>-<commit>
    [[ "$1" =~ $RELEASE_RE ]] || die "not a release id: '$1' (want <YYYY.MM.DD>-<commit>)"
}

_snapshot_path() { printf '%s/database/releases/%s.json' "$SNAPSHOT_ROOT" "$1"; }


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

_release_dir_ready() {  # <release> <dry> -> pull from Taiga when absent, then verify checksums
    local release="$1" dry="${2:-0}" dest="$ARCHIVE_ROOT/$1"
    if ! docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c "[ -f /archive/$release/SHA256SUMS ]"; then
        ssh "$TAIGA_HOST" "[ -f $TAIGA_DEST/$release/SHA256SUMS ]" || { echo "Taiga has no archived $release either"; exit 1; }
        if (( dry )); then
            echo "dry run: $release is not under $ARCHIVE_ROOT; the deploy pulls it from $TAIGA_HOST:$TAIGA_DEST/$release"
            return 0
        fi
        echo "no archived $release under $ARCHIVE_ROOT; pulling from $TAIGA_HOST:$TAIGA_DEST/$release"
        docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c "mkdir -p /archive/$release && chmod a+rwx /archive/$release"
        rsync -rlpt --partial --info=progress2 "$TAIGA_HOST:$TAIGA_DEST/$release/" "$dest/"
    fi
    docker run --rm --entrypoint bash -v "$ARCHIVE_ROOT":/archive "$IMAGE" -c "cd /archive/$release && sha256sum -c SHA256SUMS"
}

# Restore <release>'s artifact into database <db> of the serving DBMS and check it.
# The DBMS stays online: `neo4j-admin database restore` writes a database that does not
# exist yet, `CREATE DATABASE` brings it up, and the new database inherits
# server.databases.default_to_read_only. The check (`releases verify-restore`) compares
# the restored store with release.json: release id, version, the dataset set and every
# dataset's experiment count, and the node total, so the artifact is proven to be what
# its directory says it is.
_restore_into() {  # <release> <db> <dry>
    local release="$1" db="$2" dry="${3:-0}" artifact free_gb need_gb rows query
    if _cypher system "SHOW DATABASES YIELD name RETURN name;" | grep -qx "\"$db\""; then
        echo "database $db already exists in $NEO4J_CONTAINER; drop it first or deploy another release"; exit 1
    fi
    free_gb=$(docker exec "$NEO4J_CONTAINER" df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    need_gb="${MIN_FREE_GB:-300}"
    [ "$free_gb" -ge "$need_gb" ] || { echo "/data has ${free_gb}G free, need ${need_gb}G for a restored store"; exit 1; }
    if (( dry )); then
        echo "dry run: $db is free and /data has ${free_gb}G free (need ${need_gb}G); a deploy restores $release into $db and checks it against release.json"
        return 0
    fi
    artifact=$(docker exec "$NEO4J_CONTAINER" bash -c "ls $STAGING_CONT/$release/*.backup" | head -1)
    [[ -n "$artifact" ]] || { echo "no .backup artifact under $STAGING_CONT/$release"; exit 1; }
    echo "restoring $artifact -> database $db ($(date))"
    docker exec -u neo4j "$NEO4J_CONTAINER" neo4j-admin database restore --from-path="$artifact" "$db"
    _cypher system "CREATE DATABASE \`$db\` WAIT;"
    _work
    rows="$WORK/$db.rows"
    query=$(_releases restore-check-query)
    _cypher "$db" "$query" > "$rows"
    _releases verify-restore --release-json "$ARCHIVE_ROOT/$release/release.json" --rows "$rows"
    echo "restored $db ($(date))"
}

_deploy_local() {  # <release> <pin> <dry>
    local release="$1" pin="$2" dry="$3" db
    db=$(_db_name kg "$release")
    _release_dir_ready "$release" "$dry"
    _restore_into "$release" "$db" "$dry"
    if (( dry )); then
        echo "dry run: a deploy then points latest$( (( pin )) && echo ' and pinned') at $db and runs ops.sh sync"
        return 0
    fi
    # Neo4j 5 syntax is SET DATABASE TARGET; without TARGET the server answers
    # "expected 'DRIVER', 'PASSWORD', 'PROPERTIES', 'TARGET' or 'USER'" (job 5713,
    # 2026-10-17, after the 3.0 restore had already passed its check).
    _cypher system "ALTER ALIAS latest SET DATABASE TARGET \`$db\`;"
    if (( pin )); then _cypher system "ALTER ALIAS pinned SET DATABASE TARGET \`$db\`;"; fi
    echo "latest -> $db$( (( pin )) && echo ' (pinned too)'); the previous database stays for rollback (ALTER ALIAS latest SET DATABASE TARGET <name>)"
    PYTHONWARNINGS=ignore "$PY" -m torchcell.knowledge_graphs.releases --repo "$REPO_ROOT" status --label "$(hostname -s)" 2>/dev/null | grep -vE '^INFO --|DeprecationWarning|^<frozen|^$'
    bash "$REPO_ROOT/scripts/ops.sh" sync
}

# --------------------------------------------------------------------------- archive mount
# Radiant (2026-10-13 measurements in [[scripts.kg_release]]): /mnt/delta_bbub (500 G,
# Neo4j's /data until KG 4.0) and /mnt/zhao5 (50 T, the release archive) are both Taiga
# NFS exports of one server with identical mount options, so the store can live on the
# archive's mount: a restore reads the artifact in place and writes the store once, and
# the store's size is bounded by the Taiga allocation, not the 500 G export.

_remote() {  # run one command on $SERVE_HOST, every argument quoted for its shell
    local line
    line=$(printf '%q ' "$@")
    if [[ "$SERVE_HOST" == local ]]; then
        bash -c "$line" < /dev/null
    else
        ssh -n -o BatchMode=yes -o ServerAliveInterval=60 "$SERVE_HOST" "$line"
    fi
}

_rcypher() {  # <container> <database> <statement>: cypher-shell in a container on $SERVE_HOST
    local container="$1" db="$2"
    shift 2
    _remote docker exec "$container" cypher-shell -a "$SERVE_BOLT" -u "${NEO4J_USER:-neo4j}" \
        -p "${NEO4J_PASSWORD:-torchcell}" -d "$db" --format plain "$*"
}

_show() {  # print a command line as it will run, with secret-looking values masked
    printf '%q ' "$@" | sed -E 's/((AUTH|PASSWORD|SECRET|TOKEN|KEY)[A-Za-z_]*=)[^ ]+/\1***/g'
    echo
}

# The serving container as docker inspect describes it, in the CFG_* variables a
# relaunch needs. Every bind mount must be one this script knows: /data and /logs
# (replaced by $SERVE_DATA and $SERVE_LOGS), /kg-releases (the archive), the conf
# directory, and the certificates, plugins and .env mounts (carried as they are).
# Anything else, a non-default network or a resource limit refuses, so a relaunch never
# drops part of the configuration silently. CFG_ENV is the environment the container
# sets beyond its image's own (the license acceptance on Radiant).
_read_serving_config() {  # <container>
    local c="$1" dest src rw kv network limits image_env
    _remote docker inspect "$c" > /dev/null || die "no container $c on $SERVE_HOST"
    CFG_IMAGE=$(_remote docker inspect "$c" --format '{{.Config.Image}}')
    CFG_USER=$(_remote docker inspect "$c" --format '{{.Config.User}}')
    CFG_RESTART=$(_remote docker inspect "$c" --format '{{.HostConfig.RestartPolicy.Name}}')
    CFG_STATE=$(_remote docker inspect "$c" --format '{{.State.Status}}')
    network=$(_remote docker inspect "$c" --format '{{.HostConfig.NetworkMode}}')
    limits=$(_remote docker inspect "$c" --format '{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}')
    [[ -n "$CFG_USER" ]] || die "$c sets no --user; the store's owner cannot be derived"
    [[ "$network" == bridge || "$network" == default ]] || die "$c runs on network $network; a relaunch carries the default bridge only"
    [[ "$limits" == "0 0" ]] || die "$c has memory/cpu limits ($limits); a relaunch does not carry them"
    CFG_PORTS=()
    while read -r kv; do
        if [[ -n "$kv" ]]; then CFG_PORTS+=(-p "$kv"); fi
    done < <(_remote docker inspect "$c" --format '{{range $port, $binds := .HostConfig.PortBindings}}{{range $binds}}{{if .HostIp}}{{.HostIp}}:{{end}}{{.HostPort}}:{{$port}}{{println}}{{end}}{{end}}')
    (( ${#CFG_PORTS[@]} )) || die "$c publishes no ports"
    CFG_DATA="" CFG_LOGS="" CFG_ARCHIVE="" CFG_CONF="" CFG_CONF_RW=""
    CFG_KEEP=() CFG_KEEP_RO=()
    while IFS=$'\t' read -r dest src rw; do
        [[ -n "$dest" ]] || continue
        case "$dest" in
            /data) CFG_DATA="$src" ;;
            /logs) CFG_LOGS="$src" ;;
            /kg-releases) CFG_ARCHIVE="$src" ;;
            /var/lib/neo4j/conf) CFG_CONF="$src"; CFG_CONF_RW="$rw" ;;
            /var/lib/neo4j/certificates|/plugins|/.env)
                if [[ "$rw" == true ]]; then CFG_KEEP+=(-v "$src:$dest"); else CFG_KEEP+=(-v "$src:$dest:ro"); fi
                CFG_KEEP_RO+=(-v "$src:$dest:ro") ;;
            *) die "$c mounts $src at $dest, which a relaunch does not know how to carry" ;;
        esac
    done < <(_remote docker inspect "$c" --format '{{range .Mounts}}{{.Destination}}{{"\t"}}{{.Source}}{{"\t"}}{{.RW}}{{println}}{{end}}')
    [[ -n "$CFG_CONF" ]] || die "$c mounts no conf directory at /var/lib/neo4j/conf"
    [[ -n "$CFG_DATA" && -n "$CFG_LOGS" ]] || die "$c does not mount both /data and /logs"
    image_env=$(_remote docker image inspect "$CFG_IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}')
    CFG_ENV=()
    while read -r kv; do
        if [[ -n "$kv" ]] && ! grep -qxF -- "$kv" <<<"$image_env"; then CFG_ENV+=(-e "$kv"); fi
    done < <(_remote docker inspect "$c" --format '{{range .Config.Env}}{{println .}}{{end}}')
}

# The relaunch of the serving container on the archive mount (RUN_ARGS) and the staging
# DBMS that prepares that store first (STAGE_ARGS). Staging publishes no port (every
# statement goes through docker exec), runs with 2G heap and 2G page cache beside the
# serving DBMS, and reads the conf through /conf, which the image entrypoint COPIES into
# the container before applying the memory settings, so the serving conf on the host is
# never edited. Everything it mounts besides the store and its logs is read-only.
_relaunch_args() {
    local conf_mount="$CFG_CONF:/var/lib/neo4j/conf"
    [[ "$CFG_CONF_RW" == true ]] || conf_mount="$conf_mount:ro"
    RUN_ARGS=(docker run -d --name "$SERVE_CONTAINER" --user "$CFG_USER" --restart "$CFG_RESTART"
        "${CFG_PORTS[@]}" "${CFG_ENV[@]}"
        -v "$SERVE_DATA:/data" -v "$SERVE_LOGS:/logs" -v "$TAIGA_DEST:/kg-releases:ro"
        -v "$conf_mount" "${CFG_KEEP[@]}" "$CFG_IMAGE")
    STAGE_ARGS=(docker run -d --name "$SERVE_CONTAINER-staging" --user "$CFG_USER" "${CFG_ENV[@]}"
        -e NEO4J_server_memory_pagecache_size=2G
        -e NEO4J_server_memory_heap_initial__size=2G -e NEO4J_server_memory_heap_max__size=2G
        -v "$SERVE_DATA:/data" -v "$SERVE_LOGS:/logs" -v "$TAIGA_DEST:/kg-releases:ro"
        -v "$CFG_CONF:/conf:ro" "${CFG_KEEP_RO[@]}" "$CFG_IMAGE")
}

# $SERVE_DATA and $SERVE_LOGS are owned by the uid the DBMS runs as, with the group of
# the Taiga directory above them and mode 2770. A missing one is created with sudo
# (Taiga keeps root's ownership changes; the 2026-10-12 deploy chowned its staging copy
# the same way); an existing one must already belong to that uid.
_serve_dirs_ready() {  # <dry>
    local dry="$1" uid="${CFG_USER%%:*}" dir owner gid
    for dir in "$SERVE_DATA" "$SERVE_LOGS"; do
        if _remote test -d "$dir"; then
            owner=$(_remote stat -c %u "$dir")
            [[ "$owner" == "$uid" ]] || die "$dir belongs to uid $owner; the DBMS runs as uid $uid"
            echo "   dir      $dir (uid $uid)"
        else
            gid=$(_remote stat -c %g "$(dirname "$dir")")
            if (( dry )); then
                _remote sudo -n true || die "creating $dir needs passwordless sudo on $SERVE_HOST"
                echo "   dir      $dir is created: sudo install -d -o $uid -g $gid -m 2770"
            else
                _remote sudo -n install -d -o "$uid" -g "$gid" -m 2770 "$dir"
                echo "   dir      $dir created (uid $uid, gid $gid, 2770)"
            fi
        fi
    done
}

_one_artifact() {  # <release> -> the one .backup of the archived release on $SERVE_HOST
    local found
    found=$(_remote bash -c "ls -1 $(printf '%q' "$TAIGA_DEST/$1")/*.backup")
    [[ -n "$found" && $(wc -l <<<"$found") -eq 1 ]] || die "expected one .backup under $TAIGA_DEST/$1, found: ${found:-none}"
    printf '%s\n' "$found"
}

_archived_release_json() {  # <release> -> $WORK/<release>.release.json, checked to name <release>
    local release="$1" out="$WORK/$1.release.json" named
    _remote cat "$TAIGA_DEST/$release/release.json" > "$out"
    named=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["release"])' "$out")
    [[ "$named" == "$release" ]] || die "$TAIGA_DEST/$release/release.json describes release $named"
    printf '%s\n' "$out"
}

# neo4j-admin restore in a detached helper container on $SERVE_HOST, as the serving uid:
# the artifact is read in place through the read-only /kg-releases mount and unpacked
# into a temp directory under /data, the same mount as /data/databases, so the finished
# store is moved into place rather than written a second time. Detached, so a dropped
# ssh connection does not stop it (`ssh $SERVE_HOST docker logs -f kg-restore-<db>`).
_restore_on_serve_host() {  # <release> <db> <artifact path on $SERVE_HOST>
    local release="$1" db="$2" artifact="$3" name="kg-restore-$2" tmp="/data/restore-tmp-$2" t0 status code i misses
    _remote docker inspect "$name" > /dev/null 2>&1 && die "a container $name exists on $SERVE_HOST; read its logs and remove it first"
    ON_FAIL="restore of $db stopped: ssh $SERVE_HOST docker logs $name; a partial store may remain at $SERVE_DATA/databases/$db and $SERVE_DATA/restore-tmp-$db"
    echo "restoring $release -> $SERVE_DATA/databases/$db ($(date))"
    _remote docker run -d --name "$name" --user "$CFG_USER" --entrypoint bash \
        -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
        -v "$SERVE_DATA:/data" -v "$SERVE_LOGS:/logs" -v "$TAIGA_DEST:/kg-releases:ro" \
        -v "$CFG_CONF:/var/lib/neo4j/conf:ro" "$CFG_IMAGE" \
        -c "set -e; mkdir $tmp; neo4j-admin database restore --from-path=/kg-releases/$release/$(basename "$artifact") --temp-path=$tmp $db; rmdir $tmp" > /dev/null
    t0=$(date +%s)
    misses=0
    for (( i = 1; ; i++ )); do
        sleep 10
        # a poll that does not answer (an ssh blip) is retried; five silent minutes stop the run
        status=$(_remote docker inspect "$name" --format '{{.State.Status}}' 2> /dev/null || true)
        if [[ -z "$status" ]]; then
            (( ++misses < 30 )) || die "no answer from $SERVE_HOST for 5 min while $name runs there"
            continue
        fi
        misses=0
        [[ "$status" == exited || "$status" == dead ]] && break
        if (( i % 6 == 0 )); then echo "  $name $status, $(( ($(date +%s) - t0) / 60 )) min"; fi
    done
    code=$(_remote docker inspect "$name" --format '{{.State.ExitCode}}')
    _remote docker logs "$name" 2>&1 | tail -20
    echo
    [[ "$code" == 0 ]] || die "$name exited $code"
    _remote docker rm "$name" > /dev/null
    ON_FAIL=""
    echo "restored $db in $(( $(date +%s) - t0 )) s ($(date))"
}

_wait_online() {  # <container> <database>: poll for up to 10 min until it reports online
    local c="$1" db="$2" status="" i
    for i in $(seq 1 60); do
        status=$(_rcypher "$c" system "SHOW DATABASE \`$db\` YIELD currentStatus RETURN currentStatus;" 2>/dev/null | tail -1 | tr -d '"' || true)
        if [[ "$status" == online ]]; then
            echo "$c: $db online"
            return 0
        fi
        sleep 10
    done
    echo "$c: $db is ${status:-not answering} after 10 min (ssh $SERVE_HOST docker logs $c)" >&2
    return 1
}

# Register a restored database in a running DBMS and prove it before any alias moves:
# CREATE DATABASE (IF NOT EXISTS, so a rerun resumes), the restore check against the
# archive's release.json, the committed snapshot's package pairing, then READ ONLY:
# Radiant runs with auth disabled and server.databases.default_to_read_only=false, so a
# database left writable would be a writable public graph.
_register_restored() {  # <container> <release>
    local c="$1" release="$2" db rows rj query pairing access i
    db=$(_db_name kg "$release")
    rj=$(_archived_release_json "$release")
    query=$(_releases restore-check-query)
    echo "== register $release as $db in $c"
    _rcypher "$c" system "CREATE DATABASE \`$db\` IF NOT EXISTS WAIT;"
    rows="$WORK/$db.rows"
    _rcypher "$c" "$db" "$query" > "$rows"
    _releases verify-restore --release-json "$rj" --rows "$rows"
    pairing=$(_releases pairing-statement --rows "$rows" --snapshot "$(_snapshot_path "$release")")
    if [[ -n "$pairing" ]]; then
        echo "package pairing from $(_snapshot_path "$release"): $pairing"
        _rcypher "$c" "$db" "$pairing"
    fi
    _rcypher "$c" system "ALTER DATABASE \`$db\` SET ACCESS READ ONLY;"
    for i in $(seq 1 30); do
        access=$(_rcypher "$c" system "SHOW DATABASE \`$db\` YIELD access RETURN access;" | tail -1 | tr -d '"')
        [[ "$access" == read-only ]] && break
        sleep 2
    done
    [[ "$access" == read-only ]] || die "$db reports access $access after ALTER ... READ ONLY"
    echo "$db: checked, paired, read-only"
}

# latest and torchcell (and pinned, when given) point at databases; the default
# database follows latest. dbms.setDefaultDatabase needs the old default stopped, so it
# is stopped for that one call and started again, which keeps a release `pinned` still
# names online (a few seconds without it).
_point_aliases() {  # <container> <latest db> <pinned db, empty to leave pinned alone>
    local c="$1" latest="$2" pinned="$3" old
    _rcypher "$c" system "CREATE OR REPLACE ALIAS \`latest\` FOR DATABASE \`$latest\`;"
    _rcypher "$c" system "CREATE OR REPLACE ALIAS \`torchcell\` FOR DATABASE \`$latest\`;"
    if [[ -n "$pinned" ]]; then
        _rcypher "$c" system "CREATE OR REPLACE ALIAS \`pinned\` FOR DATABASE \`$pinned\`;"
    fi
    old=$(_rcypher "$c" system "SHOW DEFAULT DATABASE YIELD name RETURN name;" | tail -1 | tr -d '"')
    if [[ "$old" != "$latest" ]]; then
        _rcypher "$c" system "STOP DATABASE \`$old\` WAIT;"
        _rcypher "$c" system "CALL dbms.setDefaultDatabase('$latest');"
        _rcypher "$c" system "START DATABASE \`$old\` WAIT;"
    fi
}

_check_serving() {  # <container> <latest release>: default, aliases and the release through `torchcell`
    local c="$1" release="$2" db got
    db=$(_db_name kg "$release")
    _wait_online "$c" "$db"
    got=$(_rcypher "$c" system "SHOW DEFAULT DATABASE YIELD name RETURN name;" | tail -1 | tr -d '"')
    [[ "$got" == "$db" ]] || die "default database is $got, expected $db"
    got=$(_rcypher "$c" torchcell "MATCH (r:KgRelease) RETURN r.release;" | tail -1 | tr -d '"')
    [[ "$got" == "$release" ]] || die "the torchcell alias reads release $got, expected $release"
    _rcypher "$c" system "SHOW DATABASES YIELD name, aliases, access, currentStatus, default RETURN name, aliases, access, currentStatus, default ORDER BY name;"
    echo "serving: default $db; torchcell and latest read release $release"
}

_deploy_archive_mount() {  # <release> <pin> <dry>
    local release="$1" pin="$2" dry="$3" db artifact bytes free_gb need_gb online=0 pinned_db=""
    db=$(_db_name kg "$release")
    need_gb="${MIN_FREE_GB:-1000}"
    _work
    echo "== deploy $release from the archive mount on $SERVE_HOST"
    _remote test -f "$TAIGA_DEST/$release/SHA256SUMS" || die "no archived $release at $SERVE_HOST:$TAIGA_DEST/$release; run kg_release.sh ship $release first"
    artifact=$(_one_artifact "$release")
    bytes=$(_remote stat -c %s "$artifact")
    echo "   archive  $artifact ($bytes bytes), read in place"
    _remote bash -c "cd $(printf '%q' "$TAIGA_DEST/$release") && sha256sum -c SHA256SUMS"
    _archived_release_json "$release" > /dev/null
    [[ -f "$(_snapshot_path "$release")" ]] || die "no committed snapshot $(_snapshot_path "$release"); set SNAPSHOT_ROOT to the checkout that holds it"
    echo "   snapshot $(_snapshot_path "$release")"
    _read_serving_config "$SERVE_CONTAINER"
    if [[ "$CFG_DATA" == "$SERVE_DATA" ]]; then
        online=1
        [[ "$CFG_STATE" == running ]] || die "$SERVE_CONTAINER is $CFG_STATE"
        [[ "$CFG_ARCHIVE" == "$TAIGA_DEST" ]] || die "$SERVE_CONTAINER mounts ${CFG_ARCHIVE:-nothing} at /kg-releases, not $TAIGA_DEST"
        if _rcypher "$SERVE_CONTAINER" system "SHOW DATABASES YIELD name RETURN name;" | grep -qx "\"$db\""; then
            die "database $db already exists in $SERVE_CONTAINER"
        fi
        echo "   serving  $SERVE_CONTAINER runs on $SERVE_DATA: $db is registered online after the restore"
    else
        echo "   serving  $SERVE_CONTAINER runs on $CFG_DATA, not $SERVE_DATA: restore only; serve-swap moves serving"
    fi
    _serve_dirs_ready "$dry"
    if _remote test -e "$SERVE_DATA/databases/$db"; then die "$SERVE_DATA/databases/$db exists already"; fi
    free_gb=$(_remote df --output=avail -BG "$(dirname "$SERVE_DATA")" | tail -1 | tr -dc 0-9)
    [[ "$free_gb" -ge "$need_gb" ]] || die "$(dirname "$SERVE_DATA") has ${free_gb}G free, need ${need_gb}G (MIN_FREE_GB)"
    echo "   space    ${free_gb}G free on $(dirname "$SERVE_DATA") (need ${need_gb}G)"
    echo "   restore  helper container kg-restore-$db as $CFG_USER from $CFG_IMAGE, temp dir $SERVE_DATA/restore-tmp-$db"
    if (( dry )); then
        echo "dry run: nothing restored"
        return 0
    fi
    _restore_on_serve_host "$release" "$db" "$artifact"
    if (( ! online )); then
        echo "next: kg_release.sh serve-swap --latest <release> [--pinned <release>] once every release it should serve is restored"
        return 0
    fi
    ON_FAIL="$db is restored on $SERVE_DATA but no alias moved, so serving is unchanged. After reading the check above: DROP DATABASE \`$db\` DESTROY DATA in $SERVE_CONTAINER to redo the deploy, or serve-point to it once it is fixed."
    _register_restored "$SERVE_CONTAINER" "$release"
    if (( pin )); then pinned_db="$db"; fi
    ON_FAIL="$db passed its check but the alias move stopped partway; point serving explicitly: bash scripts/kg_release.sh serve-point --latest <release> [--pinned <release>]"
    _point_aliases "$SERVE_CONTAINER" "$db" "$pinned_db"
    _check_serving "$SERVE_CONTAINER" "$release"
    ON_FAIL=""
    bash "$REPO_ROOT/scripts/ops.sh" sync
}

_parse_point_args() {  # --latest <release> [--pinned <release>] [--dry-run] -> P_LATEST P_PINNED P_DRY
    P_LATEST="" P_PINNED="" P_DRY=0
    while (( $# )); do
        case "$1" in
            --latest) P_LATEST="${2:-}"; shift ;;
            --pinned) P_PINNED="${2:-}"; shift ;;
            --dry-run) P_DRY=1 ;;
            *) die "unknown argument $1" ;;
        esac
        shift
    done
    [[ -n "$P_LATEST" ]] || die "--latest <release> is required"
    _release_id "$P_LATEST"
    if [[ -n "$P_PINNED" ]]; then _release_id "$P_PINNED"; fi
}

cmd_serve_swap() {
    _parse_point_args "$@"
    local latest_db pinned_db stage="$SERVE_CONTAINER-staging" kept problems=0 r db
    latest_db=$(_db_name kg "$P_LATEST")
    pinned_db=$(_db_name kg "${P_PINNED:-$P_LATEST}")
    _work
    _read_serving_config "$SERVE_CONTAINER"
    echo "== serve-swap: $SERVE_CONTAINER on $SERVE_HOST from $CFG_DATA to $SERVE_DATA"
    [[ "$CFG_DATA" != "$SERVE_DATA" ]] || die "$SERVE_CONTAINER already serves $SERVE_DATA; use deploy --from-archive-mount or serve-point"
    [[ "$CFG_STATE" == running ]] || die "$SERVE_CONTAINER is $CFG_STATE"
    if _remote docker inspect "$stage" > /dev/null 2>&1; then die "a container $stage exists; read its logs and remove it first"; fi
    for r in "$P_LATEST" ${P_PINNED:+"$P_PINNED"}; do
        db=$(_db_name kg "$r")
        if _remote test -d "$SERVE_DATA/databases/$db"; then echo "   restored $db"; else echo "   NOT restored: $db (kg_release.sh deploy $r --from-archive-mount)"; problems=1; fi
        if [[ -f "$(_snapshot_path "$r")" ]]; then echo "   snapshot $(_snapshot_path "$r")"; else echo "   NO snapshot $(_snapshot_path "$r")"; problems=1; fi
        if _remote test -f "$TAIGA_DEST/$r/release.json"; then echo "   archive  $TAIGA_DEST/$r/release.json"; else echo "   NO archive $TAIGA_DEST/$r/release.json"; problems=1; fi
    done
    _relaunch_args
    echo "   aliases  latest, torchcell, default -> $latest_db; pinned -> $pinned_db"
    echo "   staging: $(_show "${STAGE_ARGS[@]}")"
    echo "   relaunch: $(_show "${RUN_ARGS[@]}")"
    echo "   kept:    $SERVE_CONTAINER is stopped and renamed $SERVE_CONTAINER-pre-<timestamp> (restart=no), store $CFG_DATA untouched"
    (( ! problems )) || die "serve-swap preconditions failed"
    if (( P_DRY )); then
        echo "dry run: nothing started or stopped"
        return 0
    fi
    ON_FAIL="serve-swap stopped while staging: $SERVE_CONTAINER still serves $CFG_DATA, untouched. Read ssh $SERVE_HOST docker logs $stage, then docker rm -f $stage; a rerun resumes (registration is idempotent)."
    _remote "${STAGE_ARGS[@]}" > /dev/null
    _wait_online "$stage" system
    _register_restored "$stage" "$P_LATEST"
    if [[ -n "$P_PINNED" && "$P_PINNED" != "$P_LATEST" ]]; then _register_restored "$stage" "$P_PINNED"; fi
    _point_aliases "$stage" "$latest_db" "$pinned_db"
    _check_serving "$stage" "$P_LATEST"
    _remote docker stop --time 300 "$stage" > /dev/null
    _remote docker rm "$stage" > /dev/null
    kept="$SERVE_CONTAINER-pre-$(date +%Y%m%d-%H%M%S)"
    ON_FAIL="serve-swap stopped during the swap; to serve the old store again: bash scripts/kg_release.sh serve-rollback $kept"
    echo "== swap ($(date))"
    _remote docker stop --time 300 "$SERVE_CONTAINER" > /dev/null
    _remote docker rename "$SERVE_CONTAINER" "$kept"
    _remote docker update --restart=no "$kept" > /dev/null
    _remote "${RUN_ARGS[@]}" > /dev/null
    _wait_online "$SERVE_CONTAINER" system
    _check_serving "$SERVE_CONTAINER" "$P_LATEST"
    ON_FAIL=""
    echo "swapped ($(date)); kept $kept, stopped, on $CFG_DATA. Rollback: bash scripts/kg_release.sh serve-rollback $kept"
    bash "$REPO_ROOT/scripts/ops.sh" sync
}

cmd_serve_point() {
    _parse_point_args "$@"
    local latest_db pinned_db="" names r
    latest_db=$(_db_name kg "$P_LATEST")
    if [[ -n "$P_PINNED" ]]; then pinned_db=$(_db_name kg "$P_PINNED"); fi
    _read_serving_config "$SERVE_CONTAINER"
    [[ "$CFG_DATA" == "$SERVE_DATA" ]] || die "serve-point acts on the archive-mount DBMS; $SERVE_CONTAINER runs on $CFG_DATA"
    names=$(_rcypher "$SERVE_CONTAINER" system "SHOW DATABASES YIELD name, currentStatus RETURN name, currentStatus;")
    for r in "$latest_db" ${pinned_db:+"$pinned_db"}; do
        grep -qx "\"$r\", \"online\"" <<<"$names" || die "$r is not an online database of $SERVE_CONTAINER"
    done
    echo "== serve-point: latest, torchcell, default -> $latest_db${pinned_db:+; pinned -> $pinned_db}"
    if (( P_DRY )); then
        echo "dry run: nothing pointed"
        return 0
    fi
    _point_aliases "$SERVE_CONTAINER" "$latest_db" "$pinned_db"
    _check_serving "$SERVE_CONTAINER" "$P_LATEST"
    bash "$REPO_ROOT/scripts/ops.sh" sync
}

cmd_serve_rollback() {
    local kept="" dry=0 aside default_db kept_data
    while (( $# )); do
        case "$1" in
            --dry-run) dry=1 ;;
            -*) die "unknown option $1" ;;
            *) kept="$1" ;;
        esac
        shift
    done
    [[ -n "$kept" ]] || die "usage: $0 serve-rollback <kept container> [--dry-run]"
    _remote docker inspect "$kept" > /dev/null 2>&1 || die "no container $kept on $SERVE_HOST"
    [[ "$(_remote docker inspect "$kept" --format '{{.State.Status}}')" == exited ]] || die "$kept is not stopped"
    kept_data=$(_remote docker inspect "$kept" --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Source}}{{end}}{{end}}')
    _read_serving_config "$SERVE_CONTAINER"
    aside="$SERVE_CONTAINER-archive-mount-$(date +%Y%m%d-%H%M%S)"
    echo "== serve-rollback on $SERVE_HOST: $SERVE_CONTAINER ($CFG_DATA) -> $aside (stopped, restart=no); $kept ($kept_data) -> $SERVE_CONTAINER (restart=$SERVE_RESTART)"
    if (( dry )); then
        echo "dry run: nothing stopped or started"
        return 0
    fi
    _remote docker stop --time 300 "$SERVE_CONTAINER" > /dev/null
    _remote docker rename "$SERVE_CONTAINER" "$aside"
    _remote docker update --restart=no "$aside" > /dev/null
    _remote docker rename "$kept" "$SERVE_CONTAINER"
    _remote docker update --restart="$SERVE_RESTART" "$SERVE_CONTAINER" > /dev/null
    _remote docker start "$SERVE_CONTAINER" > /dev/null
    _wait_online "$SERVE_CONTAINER" system
    default_db=$(_rcypher "$SERVE_CONTAINER" system "SHOW DEFAULT DATABASE YIELD name RETURN name;" | tail -1 | tr -d '"')
    _wait_online "$SERVE_CONTAINER" "$default_db"
    _rcypher "$SERVE_CONTAINER" system "SHOW DATABASES YIELD name, aliases, access, currentStatus, default RETURN name, aliases, access, currentStatus, default ORDER BY name;"
    echo "rolled back: $SERVE_CONTAINER serves $kept_data again (default $default_db); the archive-mount container is $aside, its store on $SERVE_DATA untouched"
    bash "$REPO_ROOT/scripts/ops.sh" sync
}

cmd_deploy() {
    local release="" pin=0 dry=0 archive_mount=0
    while (( $# )); do
        case "$1" in
            --pin) pin=1 ;;
            --dry-run) dry=1 ;;
            --from-archive-mount) archive_mount=1 ;;
            -*) die "unknown option $1; usage: $0 deploy <release> [--from-archive-mount] [--pin] [--dry-run]" ;;
            *) [[ -z "$release" ]] || die "one release per deploy"; release="$1" ;;
        esac
        shift
    done
    [[ -n "$release" ]] || die "usage: $0 deploy <release> [--from-archive-mount] [--pin] [--dry-run]"
    _release_id "$release"
    if (( archive_mount )); then
        _deploy_archive_mount "$release" "$pin" "$dry"
    else
        _deploy_local "$release" "$pin" "$dry"
    fi
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
    serve-swap)     shift; cmd_serve_swap "$@" ;;
    serve-point)    shift; cmd_serve_point "$@" ;;
    serve-rollback) shift; cmd_serve_rollback "$@" ;;
    restore-test) shift; cmd_restore_test "$@" ;;
    retag)   shift; cmd_retag "$@" ;;
    *) awk 'NR > 1 && /^set -euo pipefail/ {exit} NR > 1' "$0" >&2; exit 2 ;;
esac
