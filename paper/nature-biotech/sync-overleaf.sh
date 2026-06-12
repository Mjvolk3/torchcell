#!/usr/bin/env bash
# paper/nature-biotech/sync-overleaf.sh
#
# Selectively publish the SHARED (colleague-facing) manuscript from the WORKSHOP
# copy (this torchcell dir) into the Overleaf clone, then commit + push.
#
#   Workshop (private, full) = paper/nature-biotech/ in torchcell   <- you edit here
#   Shared   (Overleaf)      = $OVERLEAF_DIR                         <- colleagues see this
#
# Only the files listed below are published, so you control exactly what crosses
# over. submission.tex is published AS main.tex (Overleaf auto-detects the main
# document). Workshop-only files (editing.tex, figure-proto.tex, *.pdf, READMEs,
# this script, sn-article.tex) are intentionally NOT shared.
#
# Usage:  bash paper/nature-biotech/sync-overleaf.sh
# Override the target:  OVERLEAF_DIR=/path/to/clone bash .../sync-overleaf.sh

set -euo pipefail
export GIT_TERMINAL_PROMPT=0   # fail fast instead of hanging if a credential is missing

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${OVERLEAF_DIR:-$HOME/Documents/projects/torchcell-overleaf}"

# --- the selective share list (edit to control what colleagues get) ---
SHARE_FILES=(
  content.tex
  preamble.tex
  sn-jnl.cls
  sn-nature.bst
  references.bib
)
# submission.tex -> main.tex (handled below); figures/ copied whole.

[ -d "$DST/.git" ] || { echo "ERROR: $DST is not a git repo. Clone the Overleaf project there first." >&2; exit 1; }

# Pull collaborator changes first (images/comments they added in Overleaf), so our
# push fast-forwards and never clobbers their additions. cp below overlays only the
# files we manage; collaborator-added files are left untouched.
echo "Pulling collaborator changes from Overleaf..."
git -C "$DST" pull --no-edit

echo "Publishing shared manuscript:"
echo "  from (workshop): $SRC"
echo "  to   (Overleaf): $DST"

cp "$SRC/submission.tex" "$DST/main.tex"      # submission build = the shared main document
for f in "${SHARE_FILES[@]}"; do cp "$SRC/$f" "$DST/$f"; done
mkdir -p "$DST/figures"
cp -r "$SRC/figures/." "$DST/figures/"        # figure assets + the figure-prep guide (figures/README.md)
[ -f "$SRC/figure-proto.pdf" ] && cp "$SRC/figure-proto.pdf" "$DST/figures/figure-proto.pdf"  # true-scale sizing canvas for collaborators

cd "$DST"
git add -A
if git diff --cached --quiet; then
  echo "Nothing changed; Overleaf already up to date."
  exit 0
fi
git commit -m "Publish manuscript from torchcell workshop ($(date -u +%Y-%m-%dT%H:%MZ))"
git push
echo "Done. Pushed to Overleaf -- colleagues will see the update on next reload."
