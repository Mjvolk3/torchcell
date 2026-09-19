#!/usr/bin/env bash
# scripts/ops.sh
# [[scripts.ops]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/ops.sh
#
# Ops control panel for the torchcell services, run from GilaHyper:
#
#   make ops            full status: served knowledge-graph releases on every host,
#                       then health probes -- the default action
#   make ops-health     health probes only
#   bash scripts/ops.sh releases      the release table only
#
# The release table is one row per served database per host, in the shape the
# iBioFoundry `make ops` uses for its deployments:
#   VERSION   <major>.<minor> from the store's KgRelease node (full build bumps major,
#             incremental admission bumps minor)
#   RELEASE   <build date>-<commit[:8]>, the immutable identity a dump/backup carries
#   COMMIT#   the generation commit's index on linear main (git rev-list --count),
#             DERIVED from the release's commit; (off-main)/(unknown) when it is not
#   DATE      that commit's committer date (code recency, not build date)
#   DATASETS  Dataset nodes; NODES the count store; ALIASES which of latest/pinned
#             point here; STATUS online, or "faulting (<error>)" when a store read
#             raises (a count query still answers on a store whose pages fault, so
#             the probe reads a property)
# Then a sync verdict between the hosts and how far each served commit lags local main.
#
# Hosts (env knobs, all optional; .env is sourced first for NEO4J_* and TC_LIT_*):
#   NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD    the GilaHyper store (bolt://localhost:7687)
#   OPS_RADIANT_URI                            default neo4j+s://torchcell-database.ncsa.illinois.edu:7687
#   OPS_RADIANT_USER / OPS_RADIANT_PASSWORD    default torchcell / torchcell
#   OPS_TC_LIT_URL                             default http://localhost:8723
#   OPS_BROWSER_URL                            default http://localhost:7474
#   OPS_TIMEOUT_SECONDS                        default 5 (curl); the bolt probes get 4x
#
# Exit 0 always: a read-only reporter.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${OPS_PYTHON:-$HOME/miniconda3/envs/torchcell/bin/python}"
[[ -x "$PY" ]] || PY=python3

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

if [[ -t 1 ]]; then
    GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; YELLOW=$'\033[0;33m'; RESET=$'\033[0m'
else
    GREEN=""; RED=""; YELLOW=""; RESET=""
fi

GH_URI="${NEO4J_URI:-bolt://localhost:7687}"
GH_USER="${NEO4J_USER:-neo4j}"
GH_PASSWORD="${NEO4J_PASSWORD:-torchcell}"
RADIANT_URI="${OPS_RADIANT_URI:-neo4j+s://torchcell-database.ncsa.illinois.edu:7687}"
RADIANT_USER="${OPS_RADIANT_USER:-torchcell}"
RADIANT_PASSWORD="${OPS_RADIANT_PASSWORD:-torchcell}"
RADIANT_HTTPS="${OPS_RADIANT_HTTPS:-https://torchcell-database.ncsa.illinois.edu:7473}"
TC_LIT_URL="${OPS_TC_LIT_URL:-http://localhost:8723}"
BROWSER_URL="${OPS_BROWSER_URL:-http://localhost:7474}"
TIMEOUT="${OPS_TIMEOUT_SECONDS:-5}"
BOLT_TIMEOUT=$((TIMEOUT * 4))

line() {  # icon color name code detail
    printf "  %s%s%s  %-22s %-6s %s\n" "$2" "$1" "$RESET" "$3" "$4" "$5"
}

probe_http() {  # name url [curl args]
    local name="$1" url="$2"; shift 2
    local code
    code=$(curl --silent --output /dev/null --max-time "$TIMEOUT" --write-out "%{http_code}" "$@" "$url" 2>/dev/null)
    code="${code:-000}"
    case "$code" in
        2*|3*|4*) line "✓" "$GREEN" "$name" "$code" "$url" ;;
        5*)       line "!" "$YELLOW" "$name" "$code" "$url" ;;
        *)        line "✗" "$RED" "$name" "$code" "$url" ;;
    esac
}

# The served Browser page must carry the styling seed the image was built with.
probe_browser() {
    local body
    body=$(curl --silent --max-time "$TIMEOUT" "$BROWSER_URL/browser/" 2>/dev/null)
    if [[ -z "$body" ]]; then
        line "✗" "$RED" "neo4j browser" "000" "$BROWSER_URL/browser/"
    elif grep -q 'torchcell-seed.js' <<<"$body"; then
        line "✓" "$GREEN" "neo4j browser" "200" "$BROWSER_URL/browser/ (styling seed present)"
    else
        line "!" "$YELLOW" "neo4j browser" "200" "$BROWSER_URL/browser/ (NO styling seed: unseeded image?)"
    fi
}

# Mirrors iBioFoundry's probe: loop-status stats one heartbeat file, never the DB.
probe_merge_queue_loop() {
    local out rc state age
    out=$(timeout "$TIMEOUT" "$PY" "$REPO_ROOT/scripts/merge_queue.py" loop-status --json 2>&1)
    rc=$?
    state=$(jq -r '.state // empty' <<<"$out" 2>/dev/null)
    age=$(jq -r '.age_s // empty' <<<"$out" 2>/dev/null)
    if [[ $rc -eq 124 ]]; then
        line "✗" "$RED" "merge-queue loop" "HANG" "loop-status timed out after ${TIMEOUT}s"
    elif [[ "$state" == "live" ]]; then
        line "✓" "$GREEN" "merge-queue loop" "200" "heartbeat ${age}s ago"
    elif [[ "$state" == "stopped" ]]; then
        line "✗" "$RED" "merge-queue loop" "STOP" "heartbeat ${age:-never}s ago -- landings are NOT draining"
    else
        line "✗" "$RED" "merge-queue loop" "ERR" "${out%%$'\n'*}"
    fi
}

probe_slurm() {
    local running pending
    if ! timeout "$TIMEOUT" sinfo -h >/dev/null 2>&1; then
        line "✗" "$RED" "slurm" "000" "sinfo did not answer"
        return
    fi
    running=$(squeue -h -t R 2>/dev/null | wc -l)
    pending=$(squeue -h -t PD 2>/dev/null | wc -l)
    line "✓" "$GREEN" "slurm" "200" "$running running, $pending pending"
}

probe_disk() {  # name path
    local pct avail
    if ! df -h "$2" >/dev/null 2>&1; then
        line "✗" "$RED" "$1" "ERR" "$2 not mounted"
        return
    fi
    pct=$(df -h --output=pcent "$2" | tail -1 | tr -d ' %')
    avail=$(df -h --output=avail "$2" | tail -1 | tr -d ' ')
    if (( pct >= 90 )); then
        line "!" "$YELLOW" "$1" "${pct}%" "$2  ${avail} free"
    else
        line "✓" "$GREEN" "$1" "${pct}%" "$2  ${avail} free"
    fi
}

# The release table for both hosts in ONE python call, so the columns align. Importing
# torchcell.knowledge_graphs pulls in BioCypher, which logs a banner and a deprecation
# warning or two on import; those lines are dropped so the table stays a table. A host
# that hangs or refuses gets one "(dbms) ... faulting" row from the module itself.
release_table() {
    PYTHONWARNINGS=ignore timeout $((BOLT_TIMEOUT * 2 + 10)) "$PY" -m torchcell.knowledge_graphs.releases \
        --repo "$REPO_ROOT" status --host-timeout "$BOLT_TIMEOUT" \
        --host "gilahyper=$GH_URI|$GH_USER|$GH_PASSWORD" \
        --host "radiant=$RADIANT_URI|$RADIANT_USER|$RADIANT_PASSWORD" 2>&1 \
        | grep -vE '^INFO --|DeprecationWarning|^<frozen|^$'
}

# The release id of a host's [default] row: the <date>-<sha> token, wherever the
# columns put it (the DATABASE column can contain a space).
default_release() {  # label table
    grep -E "^$1 " <<<"$2" | grep '\[default\]' | grep -oE '[0-9]{4}\.[0-9]{2}\.[0-9]{2}-[0-9a-f]{7,}' | head -1
}

print_releases() {
    echo "== knowledge graph releases =="
    local table
    table=$(release_table)
    printf '%s\n' "$table"
    echo
    local gh_rel radiant_rel
    gh_rel=$(default_release gilahyper "$table")
    radiant_rel=$(default_release radiant "$table")
    if [[ -n "$gh_rel" && "$gh_rel" != "-" && "$gh_rel" == "$radiant_rel" ]]; then
        echo "sync: in sync ($gh_rel)"
    else
        echo "sync: DIVERGED -- gilahyper=${gh_rel:-?} radiant=${radiant_rel:-?}"
    fi
    local main_sha main_date main_subject
    main_sha=$(git -C "$REPO_ROOT" rev-parse --short=8 main 2>/dev/null || echo "(no main)")
    main_date=$(git -C "$REPO_ROOT" show -s --format=%cd --date=format:%Y.%m.%d main 2>/dev/null || echo "????.??.??")
    main_subject=$(git -C "$REPO_ROOT" log -1 --format=%s main 2>/dev/null)
    echo "main: ${main_date}-${main_sha}${main_subject:+  (${main_subject})}"
    for pair in "gilahyper=$gh_rel" "radiant=$radiant_rel"; do
        local host="${pair%%=*}" rel="${pair#*=}" sha behind
        if [[ -z "$rel" || "$rel" == "-" ]]; then
            printf '      %s: no release node\n' "$host"; continue
        fi
        sha="${rel##*-}"
        if ! git -C "$REPO_ROOT" cat-file -e "${sha}^{commit}" 2>/dev/null; then
            printf '      %s: commit %s not in local history\n' "$host" "$sha"; continue
        fi
        behind=$(git -C "$REPO_ROOT" rev-list --count "${sha}..main" 2>/dev/null || echo "?")
        if [[ "$behind" == "0" ]]; then
            printf '      %s: up to date with main\n' "$host"
        else
            printf '      %s: %s behind main\n' "$host" "$behind"
        fi
    done
}

print_health() {
    echo "== health (gilahyper) =="
    probe_browser
    probe_http "tc-lit" "$TC_LIT_URL/health"
    probe_merge_queue_loop
    probe_slurm
    probe_disk "disk /scratch" "/scratch"
    probe_disk "disk /db" "/db"
    probe_disk "disk /bulk" "/bulk"
    echo "== health (radiant) =="
    probe_http "radiant https" "$RADIANT_HTTPS/" -k
}

ACTION="${1:-status}"
case "$ACTION" in
    status)   print_releases; echo; print_health ;;
    releases) print_releases ;;
    health)   print_health ;;
    *)
        echo "usage: $0 {status|releases|health}" >&2
        exit 2
        ;;
esac
