#!/usr/bin/env bash
# paper/nature-biotech/sync-overleaf.sh
#
# Selectively publish the SHARED (colleague-facing) manuscript from the WORKSHOP
# copy (this torchcell dir) into the Overleaf clone, then commit + push.
#
#   Workshop (private, full) = paper/nature-biotech/ in torchcell   <- you edit here
#   Shared   (Overleaf)      = $OVERLEAF_DIR                         <- colleagues see this
#
# submission.tex is published AS main.tex (Overleaf's default main document);
# editing.tex and twocolumn.tex are shared as alternate views (switch Menu ->
# Main document in Overleaf). sections/ and figures/ are copied whole. Workshop-
# only files (figure-proto.tex, *.pdf views, READMEs, this script, sn-article.tex,
# flatten_tex.py, Makefile) are intentionally NOT shared.
#
# DELETION POLICY: this propagates OUR OWN deletions (a file we previously
# published but no longer do is removed from Overleaf) while NEVER touching
# collaborator-added files. It does this via a manifest of files we publish,
# kept out of the Overleaf push (.git/info/exclude). The pull-before-push also
# preserves any images/comments collaborators added in Overleaf.
#
# Usage:  bash paper/nature-biotech/sync-overleaf.sh
# Override the target:  OVERLEAF_DIR=/path/to/clone bash .../sync-overleaf.sh

set -euo pipefail
export GIT_TERMINAL_PROMPT=0   # fail fast instead of hanging if a credential is missing

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${OVERLEAF_DIR:-$HOME/Documents/projects/torchcell-overleaf}"
MANIFEST=".torchcell-sync-manifest"   # relative to DST; records files WE publish

# --- the selective share list (edit to control what colleagues get) ---
SHARE_FILES=(
  content.tex
  preamble.tex
  sn-jnl.cls
  sn-nature.bst
  references.bib
  editing.tex
  twocolumn.tex
)
# submission.tex -> main.tex (handled below); sections/ and figures/ copied whole.

[ -d "$DST/.git" ] || { echo "ERROR: $DST is not a git repo. Clone the Overleaf project there first." >&2; exit 1; }

# Keep our manifest out of the Overleaf push (local-only bookkeeping).
grep -qxF "$MANIFEST" "$DST/.git/info/exclude" 2>/dev/null || echo "$MANIFEST" >> "$DST/.git/info/exclude"

# Pull collaborator changes first (images/comments they added in Overleaf).
echo "Pulling collaborator changes from Overleaf..."
git -C "$DST" pull --no-edit

# The exact set of files we publish this run (paths relative to DST), sorted.
current_files() {
  echo "main.tex"
  printf '%s\n' "${SHARE_FILES[@]}"
  for f in "$SRC"/sections/*.tex; do [ -e "$f" ] && echo "sections/$(basename "$f")"; done
  [ -d "$SRC/figures" ] && ( cd "$SRC/figures" && find . -type f | sed 's|^\./|figures/|' )
  [ -f "$SRC/figure-proto.pdf" ] && echo "figures/figure-proto.pdf"
}
CUR="$(current_files | sort -u)"

# Propagate OUR deletions: anything in the previous manifest but not in CUR is a
# file we dropped -> remove it. Collaborator files were never in the manifest.
if [ -f "$DST/$MANIFEST" ]; then
  while read -r gone; do
    [ -n "$gone" ] || continue
    if [ -e "$DST/$gone" ]; then
      git -C "$DST" rm -q --ignore-unmatch -- "$gone" >/dev/null 2>&1 || rm -f "$DST/$gone"
      echo "  removed (no longer published): $gone"
    fi
  done < <(comm -23 <(sort -u "$DST/$MANIFEST") <(printf '%s\n' "$CUR"))
fi

# Publish current files.
echo "Publishing shared manuscript:"
echo "  from (workshop): $SRC"
echo "  to   (Overleaf): $DST"
cp "$SRC/submission.tex" "$DST/main.tex"
for f in "${SHARE_FILES[@]}"; do cp "$SRC/$f" "$DST/$f"; done
mkdir -p "$DST/figures" && cp -r "$SRC/figures/." "$DST/figures/"
[ -f "$SRC/figure-proto.pdf" ] && cp "$SRC/figure-proto.pdf" "$DST/figures/figure-proto.pdf"
mkdir -p "$DST/sections" && cp "$SRC"/sections/*.tex "$DST/sections/"

# Record what we published (local-only, excluded from the push).
printf '%s\n' "$CUR" > "$DST/$MANIFEST"

cd "$DST"
git add -A
if git diff --cached --quiet; then
  echo "Nothing changed; Overleaf already up to date."
  exit 0
fi
git commit -m "Publish manuscript from torchcell workshop ($(date -u +%Y-%m-%dT%H:%MZ))"
git push
echo "Done. Pushed to Overleaf -- colleagues will see the update on next reload."
