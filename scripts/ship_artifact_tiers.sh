#!/usr/bin/env bash
# scripts/ship_artifact_tiers.sh
# [[scripts.ship_artifact_tiers]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/ship_artifact_tiers.sh
#
# Ship the file tiers tc-data serves (the raw mirror, the genomes tier, the objects tier)
# from this host to Taiga, where the database lives, and verify every manifest-listed
# file by sha256 on arrival. Radiant's tc-data reads the tiers from the Taiga mount
# (TC_DATA_RAW_ROOT, TC_DATA_GENOMES_ROOT, TC_DATA_OBJECTS_ROOT in tc-data.conf), so a
# shipped tier is served the moment it lands; the resolver on any client
# (torchcell.artifacts.resolve) then reaches it over HTTP.
#
#   ship_artifact_tiers.sh [--dry-run] [tier ...]    default: genomes objects raw
#
# For each tier: a SHA256SUMS file per key directory is written from that key's
# manifest.json (the manifest is the authority on what a key holds; the file lists
# exactly the manifest's paths), the key directories are rsynced with -rlpt (no owner or
# group, Taiga's setgid directories refuse chgrp; no --delete, the archive is a backstop),
# and `sha256sum -c SHA256SUMS` runs on Taiga for every key. Any failed check fails the
# ship. A tier directory that does not exist locally is skipped with a line, not an
# error, so the objects tier can be shipped before its first deposit exists.
#
# Env: DATA_ROOT (from .env), TAIGA_HOST (rocky@141.142.216.218),
#      TAIGA_DATA (/mnt/zhao5/mjvolk3/projects/torchcell/data/torchcell),
#      PY (the torchcell interpreter).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/torchcell/bin/python}"
TAIGA_HOST="${TAIGA_HOST:-rocky@141.142.216.218}"
TAIGA_DATA="${TAIGA_DATA:-/mnt/zhao5/mjvolk3/projects/torchcell/data/torchcell}"
if [[ -z "${DATA_ROOT:-}" && -f "$REPO_ROOT/.env" ]]; then
    DATA_ROOT=$(grep -E '^DATA_ROOT=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d '"')
fi
[[ -n "${DATA_ROOT:-}" ]] || { echo "DATA_ROOT is not set" >&2; exit 2; }

dry=()
if [[ "${1:-}" == "--dry-run" ]]; then dry=(--dry-run --itemize-changes); shift; fi
tiers=("$@"); [[ ${#tiers[@]} -gt 0 ]] || tiers=(genomes objects raw)

tier_dir() {  # <tier> -> local directory name
    case "$1" in
        raw) echo torchcell-raw ;;
        genomes) echo torchcell-genomes ;;
        objects) echo torchcell-objects ;;
        *) echo "unknown tier $1 (raw|genomes|objects)" >&2; exit 2 ;;
    esac
}

# SHA256SUMS per key from manifest.json: the manifest's files, nothing else.
write_sums() {  # <tier root>
    "$PY" - "$1" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
n_keys = n_files = 0
for manifest in sorted(root.glob("*/manifest.json")):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    lines = [f"{f['sha256']}  {f['path']}" for f in data["files"]]
    (manifest.parent / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_keys += 1
    n_files += len(lines)
print(f"  SHA256SUMS written for {n_keys} keys, {n_files} files")
EOF
}

status=0
for tier in "${tiers[@]}"; do
    name=$(tier_dir "$tier")
    src="$DATA_ROOT/$name"
    if [[ ! -d "$src" ]]; then echo "-- $tier: no $src, skipped"; continue; fi
    echo "-- $tier: $src -> $TAIGA_HOST:$TAIGA_DATA/$name"
    write_sums "$src"
    ssh "$TAIGA_HOST" "mkdir -p $TAIGA_DATA/$name"
    rsync -rlpt --partial --info=progress2 "${dry[@]}" "$src/" "$TAIGA_HOST:$TAIGA_DATA/$name/"
    if [[ ${#dry[@]} -gt 0 ]]; then continue; fi
    # every key: sha256sum -c on Taiga; a key without SHA256SUMS has no manifest and is reported
    # find, not a glob: the login shell on the far side may be zsh, whose unmatched */ is an error
    if ! ssh "$TAIGA_HOST" "cd $TAIGA_DATA/$name && rc=0; for d in \$(find . -mindepth 1 -maxdepth 1 -type d | sort); do d=\${d#./}; if [ -f \"\$d/SHA256SUMS\" ]; then (cd \"\$d\" && sha256sum -c --quiet SHA256SUMS) && echo \"  OK \$d\" || { echo \"  FAILED \$d\"; rc=1; }; else echo \"  (no manifest) \$d\"; fi; done; exit \$rc"; then
        echo "$tier: sha256 check FAILED on Taiga" >&2; status=1
    fi
done
echo "ship_artifact_tiers done $(date -Is), exit $status"
exit $status
