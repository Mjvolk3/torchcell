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

case "${1:-}" in
    backup)  cmd_backup ;;
    archive) cmd_archive ;;
    ship)    shift; cmd_ship "$@" ;;
    keep)    shift; cmd_keep "$@" ;;
    list)    cmd_list ;;
    purge)   shift; cmd_purge "$@" ;;
    *) sed -n '2,40p' "$0" >&2; exit 2 ;;
esac
