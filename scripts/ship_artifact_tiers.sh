#!/usr/bin/env bash
# scripts/ship_artifact_tiers.sh
# [[scripts.ship_artifact_tiers]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/ship_artifact_tiers.sh
#
# Ship the file tiers tc-data serves (the raw mirror, the genomes tier, the objects tier,
# the public part of the library tier) from this host to Taiga, where the database
# lives, and verify every shipped file by sha256 on arrival. Radiant's tc-data reads the
# tiers from the Taiga mount (TC_DATA_RAW_ROOT, TC_DATA_GENOMES_ROOT,
# TC_DATA_OBJECTS_ROOT, TC_DATA_LIBRARY_ROOT in tc-data.conf), so a shipped tier is
# served the moment it lands; the resolver on any client (torchcell.artifacts.resolve)
# then reaches it over HTTP.
#
#   ship_artifact_tiers.sh [--dry-run] [tier ...]    default: genomes objects raw library
#
# raw, genomes, objects: a SHA256SUMS file per key directory is written from that key's
# manifest.json (the manifest is the authority on what a key holds; the file lists
# exactly the manifest's paths), the whole tier directory is rsynced with -rlpt (no
# owner or group, Taiga's setgid directories refuse chgrp; no --delete, the archive is a
# backstop).
#
# library: ONLY the public files leave this host. A key's public records are the
# manifest records whose role does not start with paper_ or ocr_ (the same rule as
# WITHHELD_ROLE_PREFIXES in torchcell/datasets/server.py), read through the literature
# Manifest model so a manifest tc-data would refuse also stops the ship. SHA256SUMS
# lists only those files, and rsync runs with an explicit --files-from list (every
# public file, each key's manifest.json, each key's SHA256SUMS), never the directory, so
# no paper PDF, OCR markdown, OCR image or layout file is ever sent. A key with no public
# file gets no SHA256SUMS and ships only its manifest.json; underscore-prefixed service
# directories (_bib) are not keys and are not shipped.
#
# Then `sha256sum -c SHA256SUMS` runs on Taiga for every key (a library key without one
# prints "(no public files)"). Any failed check fails the ship. A tier directory that
# does not exist locally is skipped with a line, not an error, so the objects tier can
# be shipped before its first deposit exists. --dry-run passes --dry-run to rsync (the
# SHA256SUMS files and the library file list are still written locally) and skips the
# remote check.
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
tiers=("$@"); [[ ${#tiers[@]} -gt 0 ]] || tiers=(genomes objects raw library)

tier_dir() {  # <tier> -> local directory name
    case "$1" in
        raw) echo torchcell-raw ;;
        genomes) echo torchcell-genomes ;;
        objects) echo torchcell-objects ;;
        library) echo torchcell-library ;;
        *) echo "unknown tier $1 (raw|genomes|objects|library)" >&2; exit 2 ;;
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

# Library: SHA256SUMS of the public files only, and the rsync --files-from list.
write_public_list() {  # <tier root> <list file>
    PYTHONPATH="$REPO_ROOT" "$PY" - "$1" "$2" <<'EOF'
import sys
from pathlib import Path, PurePosixPath

from torchcell.literature.manifest import Manifest

WITHHELD_ROLE_PREFIXES = ("paper_", "ocr_")  # torchcell/datasets/server.py
root, list_file = Path(sys.argv[1]), Path(sys.argv[2])
entries: list[str] = []
n_keys = n_public = n_withheld = n_empty = 0
for manifest_path in sorted(root.glob("*/manifest.json")):
    key = manifest_path.parent.name
    if key.startswith("_"):
        continue
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    public = [f for f in manifest.files if not f.role.startswith(WITHHELD_ROLE_PREFIXES)]
    n_keys += 1
    n_withheld += len(manifest.files) - len(public)
    entries.append(f"{key}/manifest.json")
    if not public:
        n_empty += 1
        continue
    for f in public:
        rel = PurePosixPath(f.path)
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit(f"{manifest_path}: path escapes the key directory: {f.path}")
        entries.append(f"{key}/{f.path}")
    sums = [f"{f.sha256}  {f.path}" for f in public]
    (manifest_path.parent / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    entries.append(f"{key}/SHA256SUMS")
    n_public += len(public)
list_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
print(
    f"  {n_keys} keys: {n_public} public files listed, {n_withheld} paper/OCR files "
    f"withheld, {n_empty} keys with no public files"
)
EOF
}

status=0
for tier in "${tiers[@]}"; do
    name=$(tier_dir "$tier")
    src="$DATA_ROOT/$name"
    if [[ ! -d "$src" ]]; then echo "-- $tier: no $src, skipped"; continue; fi
    echo "-- $tier: $src -> $TAIGA_HOST:$TAIGA_DATA/$name"
    if [[ "$tier" == library ]]; then
        list_file=$(mktemp /tmp/ship_library_files.XXXXXX)
        write_public_list "$src" "$list_file"
        ssh "$TAIGA_HOST" "mkdir -p $TAIGA_DATA/$name"
        # --files-from implies --relative and no recursion: exactly the listed files go
        rsync -lpt --partial --info=progress2 --files-from="$list_file" "${dry[@]}" \
            "$src/" "$TAIGA_HOST:$TAIGA_DATA/$name/"
        rm -f "$list_file"
        missing_note="(no public files)"
    else
        write_sums "$src"
        ssh "$TAIGA_HOST" "mkdir -p $TAIGA_DATA/$name"
        rsync -rlpt --partial --info=progress2 "${dry[@]}" "$src/" "$TAIGA_HOST:$TAIGA_DATA/$name/"
        missing_note="(no manifest)"
    fi
    if [[ ${#dry[@]} -gt 0 ]]; then continue; fi
    # every key: sha256sum -c on Taiga; a key without SHA256SUMS is reported with $missing_note
    # find, not a glob: the login shell on the far side may be zsh, whose unmatched */ is an error
    if ! ssh "$TAIGA_HOST" "cd $TAIGA_DATA/$name && rc=0; for d in \$(find . -mindepth 1 -maxdepth 1 -type d | sort); do d=\${d#./}; if [ -f \"\$d/SHA256SUMS\" ]; then (cd \"\$d\" && sha256sum -c --quiet SHA256SUMS) && echo \"  OK \$d\" || { echo \"  FAILED \$d\"; rc=1; }; else echo \"  $missing_note \$d\"; fi; done; exit \$rc"; then
        echo "$tier: sha256 check FAILED on Taiga" >&2; status=1
    fi
done
echo "ship_artifact_tiers done $(date -Is), exit $status"
exit $status
