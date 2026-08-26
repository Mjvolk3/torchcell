#!/usr/bin/env bash
# pre-commit ontology-figure wrapper.
#
# The artifacts under $ASSET_IMAGES_DIR/schema-ontology are a pure function of the
# pydantic schema, the generator, and the palette. Nothing in them is hand-authored, so
# they should not be maintained by remembering to run a script. When an input is staged,
# regenerate and stage the result, so a commit that changes the schema carries the
# matching figure.
#
# Resolves the torchcell conda env python by $HOME-relative path (pre-commit's `entry:`
# is not shell-interpreted, so ~ cannot expand there), matching scripts/run-mypy.sh and
# scripts/run-schema-impact.sh. Regenerates the whole set, so staged filenames are not
# needed. Generator: paper/nature-biotech/scripts/generate_ontology_diagram.py
#
# EXITS NON-ZERO WHEN AN ARTIFACT CHANGED, deliberately. The regenerated files are staged
# for you and re-running `git commit` succeeds, but the failure is what tells you the
# figure moved. Silent regeneration would let the printed panel's layout degrade unseen
# as the schema grows, and only a human can judge whether it still reads well. This also
# mirrors ruff-format, which rewrites staged files and fails so the change gets seen.
set -euo pipefail

PY="$HOME/miniconda3/envs/torchcell/bin/python"

# Introspect the checkout being committed, not whatever torchcell is installed.
# `python path/to/script.py` puts the SCRIPT'S directory on sys.path, not the cwd, so from
# a worktree the generator otherwise imports the PRIMARY checkout's torchcell and
# regenerates the figure from the wrong branch's schema. Since all work goes through
# worktrees, that would be wrong nearly every time, and silently: the hook would report
# "already matches" for a schema change it never saw.
REPO_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Same resolution the generator uses, so the hook can never look in a different place.
# load_dotenv() with no argument locates .env by walking the CALLER'S stack frame, which
# has no parent when the program arrives on stdin, so it asserts. pre-commit runs hooks
# from the repo root, so name the file explicitly.
OUT_DIR="$("$PY" - <<'EOF'
import os
import os.path as osp

from dotenv import load_dotenv

load_dotenv(".env")
print(osp.join(os.environ["ASSET_IMAGES_DIR"], "schema-ontology"))
EOF
)"

ARTIFACTS=(
  torchcell-ontology.svg
  torchcell-ontology-overview.svg
  torchcell-ontology-schematic.svg
  torchcell-ontology-explorer.html
)

# Hash before and after rather than diffing against HEAD: pre-commit runs with unstaged
# changes stashed, so the working tree IS the staged state, and a pre/post comparison
# answers exactly "do the staged artifacts match the staged inputs".
hash_of() {
  if [ -f "$1" ]; then
    sha256sum "$1" | cut -d' ' -f1
  else
    echo absent
  fi
}

declare -A before
for f in "${ARTIFACTS[@]}"; do
  before["$f"]="$(hash_of "$OUT_DIR/$f")"
done

"$PY" paper/nature-biotech/scripts/generate_ontology_diagram.py >/dev/null

changed=()
for f in "${ARTIFACTS[@]}"; do
  if [ "${before[$f]}" != "$(hash_of "$OUT_DIR/$f")" ]; then
    changed+=("$f")
  fi
done

if [ ${#changed[@]} -eq 0 ]; then
  echo "ontology figure: already matches the schema"
  exit 0
fi

git add -- "$OUT_DIR"

echo
echo "ontology figure: regenerated and staged (an input to it changed)"
for f in "${changed[@]}"; do
  echo "    $f"
done
echo
echo "  The schematic is a paper figure. Open it before you re-commit:"
echo "    $OUT_DIR/torchcell-ontology-schematic.svg"
echo
echo "  Re-run 'git commit' to accept. The files are already staged."
exit 1
