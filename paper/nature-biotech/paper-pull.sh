#!/usr/bin/env bash
# paper/nature-biotech/paper-pull.sh
#
# Bring COLLABORATOR edits made in Overleaf back INTO the workshop, via a per-file
# 3-way merge. The companion to sync-overleaf.sh (which publishes workshop -> Overleaf);
# together they make managed files (content.tex, sections/*, ...) truly bidirectional.
#
#   Workshop (private, full) = paper/nature-biotech/ in torchcell   <- merges land here
#   Shared   (Overleaf)      = $OVERLEAF_DIR                         <- collaborators edit here
#
# HOW IT WORKS: for each file we publish, we 3-way merge
#     BASE   = the state we last published      (common ancestor)
#     OURS   = current workshop file
#     THEIRS = current Overleaf file            (after collaborator edits)
# Clean hunks merge silently; lines BOTH sides changed get <<<<<<< conflict markers.
# Results are written into the WORKSHOP files only -- nothing is pushed, nothing is
# auto-committed. Review with `git diff`, resolve any conflicts, commit in torchcell,
# then `make paper-sync` to republish the reconciled manuscript.
#
# BASE detection: sync-overleaf.sh records the published commit SHA in
# .torchcell-sync-base (local-only, like the manifest). If that's absent we fall back
# to the most recent "Publish manuscript from torchcell workshop" commit in Overleaf.
#
# Usage:  bash paper/nature-biotech/paper-pull.sh
# Test against sandboxes:
#   OVERLEAF_DIR=/path/to/clone WORKSHOP_DIR=/path/to/workshop-copy bash .../paper-pull.sh

set -euo pipefail
export GIT_TERMINAL_PROMPT=0   # fail fast instead of hanging if a credential is missing

SRC="${WORKSHOP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DST="${OVERLEAF_DIR:-$HOME/Documents/projects/torchcell-overleaf}"
WS_GIT="$(git -C "$SRC" rev-parse --show-toplevel 2>/dev/null || true)"  # repo holding the workshop (for the VS Code merge editor)
BASEFILE=".torchcell-sync-base"          # relative to DST; records our last-published SHA
MANIFEST=".torchcell-sync-manifest"      # relative to DST; records files WE publish
PUBLISH_MSG="Publish manuscript from torchcell workshop"

[ -d "$DST/.git" ] || { echo "ERROR: $DST is not a git repo. Clone the Overleaf project there first." >&2; exit 1; }

# 1) Update the Overleaf clone to the latest collaborator state.
echo "Fetching latest from Overleaf..."
git -C "$DST" pull --ff-only --no-edit

# 2) Resolve BASE = the state we last published (the 3-way merge's common ancestor).
BASE=""
[ -f "$DST/$BASEFILE" ] && BASE="$(cat "$DST/$BASEFILE")"
if [ -z "$BASE" ]; then
  BASE="$(git -C "$DST" log --grep="$PUBLISH_MSG" -n1 --format=%H || true)"
  [ -n "$BASE" ] && echo "(no $BASEFILE; falling back to last publish commit)"
fi
[ -n "$BASE" ] || { echo "ERROR: cannot determine the last-published base commit." >&2; exit 1; }
echo "Base (last published) = $(git -C "$DST" log -n1 --format='%h %s' "$BASE")"

# managed text files, "overleaf_path:workshop_path" (main.tex is published from submission.tex).
PAIRS=(
  "main.tex:submission.tex"
  "content.tex:content.tex"
  "preamble.tex:preamble.tex"
  "references.bib:references.bib"
  "editing.tex:editing.tex"
  "twocolumn.tex:twocolumn.tex"
  "sn-jnl.cls:sn-jnl.cls"
  "sn-nature.bst:sn-nature.bst"
)
for f in "$DST"/sections/*.tex; do
  [ -e "$f" ] || continue
  b="sections/$(basename "$f")"
  PAIRS+=("$b:$b")
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

merged_clean=(); merged_conflict=(); merged_new=(); unchanged=0; merge_editor_files=()

for pair in "${PAIRS[@]}"; do
  op="${pair%%:*}"; wp="${pair##*:}"
  theirs="$DST/$op"; ours="$SRC/$wp"
  [ -e "$theirs" ] || continue

  base="$TMP/base"
  git -C "$DST" show "$BASE:$op" > "$base" 2>/dev/null || : > "$base"

  # Collaborator made no change to this file relative to what we published -> skip.
  if cmp -s "$base" "$theirs"; then unchanged=$((unchanged+1)); continue; fi

  # Collaborator changed it. If we don't have it in the workshop, just take theirs.
  if [ ! -e "$ours" ]; then cp "$theirs" "$ours"; merged_new+=("$wp"); continue; fi

  # 3-way merge collaborator changes into the workshop copy.
  cp "$ours" "$TMP/ours_orig"   # OUR version (pre-markers) -- becomes git stage 2
  cp "$ours" "$TMP/merged"
  if git merge-file -L "workshop/$wp" -L "last-published" -L "overleaf/$op" \
        "$TMP/merged" "$base" "$theirs"; then
    cp "$TMP/merged" "$ours"; merged_clean+=("$wp")
  else
    cp "$TMP/merged" "$ours"; merged_conflict+=("$wp")
    # If the workshop is a git repo and this file is tracked, register a real
    # UNMERGED index entry (stage 1=base, 2=ours, 3=theirs) so VS Code lists it
    # under "Merge Changes" and offers "Resolve in Merge Editor" (3-pane + base).
    if [ -n "$WS_GIT" ]; then
      rel="${ours#"$WS_GIT"/}"
      if git -C "$WS_GIT" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
        bb="$(git -C "$DST" show "$BASE:$op" 2>/dev/null | git -C "$WS_GIT" hash-object -w --stdin)"
        ob="$(git -C "$WS_GIT" hash-object -w -- "$TMP/ours_orig")"
        tb="$(git -C "$WS_GIT" hash-object -w -- "$theirs")"
        git -C "$WS_GIT" update-index --force-remove -- "$rel"
        printf '100644 %s 1\t%s\n100644 %s 2\t%s\n100644 %s 3\t%s\n' \
          "$bb" "$rel" "$ob" "$rel" "$tb" "$rel" | git -C "$WS_GIT" update-index --index-info
        merge_editor_files+=("$rel")
      fi
    fi
  fi
done

# Figures are binary -- report collaborator changes, don't try to merge.
fig_changed=()
for f in "$DST"/figures/*; do
  [ -e "$f" ] || continue
  op="figures/$(basename "$f")"
  fb="$TMP/figbase"
  if git -C "$DST" show "$BASE:$op" > "$fb" 2>/dev/null; then
    cmp -s "$fb" "$f" || fig_changed+=("$op")
  fi
done

# New files collaborators added in Overleaf that we never published (kept on Overleaf).
new_files=()
if [ -f "$DST/$MANIFEST" ]; then
  while read -r f; do
    [ -n "$f" ] || continue
    grep -qxF "$f" "$DST/$MANIFEST" || new_files+=("$f")
  done < <(git -C "$DST" ls-files)
fi

echo
echo "==================== paper-pull summary ===================="
echo "Merged into workshop ($SRC):"
[ ${#merged_clean[@]}    -gt 0 ] && printf '  [clean]    %s\n' "${merged_clean[@]}"
[ ${#merged_conflict[@]} -gt 0 ] && printf '  [CONFLICT] %s\n' "${merged_conflict[@]}"
[ ${#merged_new[@]}      -gt 0 ] && printf '  [new]      %s\n' "${merged_new[@]}"
[ $((${#merged_clean[@]} + ${#merged_conflict[@]} + ${#merged_new[@]})) -eq 0 ] && echo "  (no managed files changed by collaborators)"
echo "Unchanged managed files: $unchanged"
[ ${#fig_changed[@]} -gt 0 ] && { echo "Figures changed in Overleaf (binary -- merge by hand if intended):"; printf '  %s\n' "${fig_changed[@]}"; }
[ ${#new_files[@]}   -gt 0 ] && { echo "Collaborator-added files (kept on Overleaf, not pulled):"; printf '  %s\n' "${new_files[@]}"; }
echo "============================================================"

if [ ${#merged_conflict[@]} -gt 0 ]; then
  echo
  if [ ${#merge_editor_files[@]} -gt 0 ]; then
    echo "ACTION (VS Code): open Source Control -> 'Merge Changes' -> click each file ->"
    echo "  'Resolve in Merge Editor' (Current = workshop, Incoming = overleaf, toggle Base"
    echo "  to see the last-published ancestor). Save, then 'git add' each resolved file."
    echo "  Files awaiting resolution:"; printf '    %s\n' "${merge_editor_files[@]}"
  else
    echo "ACTION: resolve <<<<<<< conflict markers in the files above (not a git repo, so no merge editor)."
  fi
  echo "  Then commit in torchcell, and 'make paper-sync' to republish."
  exit 2
fi
echo
echo "Review with 'git diff', commit in torchcell, then 'make paper-sync' to republish."
