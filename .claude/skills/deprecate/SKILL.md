---
name: deprecate
description: Move a path (dataset, LMDB build, schema, code, note) into the deprecation graveyard instead of deleting it. Use when retiring old data structures or stale builds -- especially before a rebuild -- since rm -rf of directories is blocked. Wraps scripts/deprecate.sh (move + provenance manifest); nothing is auto-deleted, you purge the graveyard by hand.
---

# Deprecate (move, don't delete)

Retire a path by **moving it into the deprecation graveyard** with a provenance manifest,
never by deleting it. This is the safe counterpart to `rm -rf` (which is blocked in this
environment): deprecated things filter into a graveyard you can watch in the VS Code
explorer, then purge on your own schedule.

**Usage:** `/deprecate <path> [reason]`

## When to use

- Clearing a **stale dataset build** before rebuilding it (the dataset base class skips
  `process()` when `processed/` exists, so a stale LMDB is silently reused unless moved).
- Retiring an **old schema / module / loader** that a refactor replaced.
- Setting aside a **note or scratch artifact** that matured or was superseded.

Do NOT use it to move something you did not create or cannot see the contents of without
first surfacing what it is and why it's being retired.

## The graveyard

- Default: `/tmp/torchcell-deprecated` (a GilaHyper path; also a folder in the VS Code
  workspace so you can see things arrive).
- Override with `DEPRECATED_DIR` for a **same-filesystem** move of large data (e.g. an LMDB
  under `$DATA_ROOT` -- keeping the move on `/scratch` avoids a cross-filesystem copy):
  `DEPRECATED_DIR=$DATA_ROOT/torchcell-deprecated /deprecate <path> "reason"`.
- **Nothing is auto-deleted.** Purging the graveyard is a manual, periodic chore (yours).

## Steps

1. **Confirm the target.** Show what it is (`ls -ld`, and for a dir a shallow `find … -maxdepth 2`
   or the record count / identity for an LMDB) so the move is auditable. For a dataset build,
   confirm it is the *stale* one (e.g. old record count / old schema) before moving.
2. **Move it.** Run the script from the repo root; pass a concrete `reason`:
   ```bash
   bash scripts/deprecate.sh <path> "<why it is being retired>"
   # large data on /scratch:
   DEPRECATED_DIR="$DATA_ROOT/torchcell-deprecated" bash scripts/deprecate.sh <path> "<why>"
   ```
   The script resolves the absolute source path, creates `"$GRAVEYARD/<timestamp>__<name>/"`,
   writes `DEPRECATION.txt` (original path, timestamp, host, user, git HEAD, size, reason),
   then `mv`s the target in.
3. **Report** the destination path and the manifest location (the script prints both). If this
   was a pre-rebuild clear, note that the original path is now free for the rebuild to
   repopulate.

## Rules

- **Move, never delete.** No `rm -rf`; that is the whole point.
- **Only the dev-writable trees.** You cannot move the KG-build tree
  (`$DATA_ROOT/database/data/...`, owned by uid 7474) -- that is the KG-build user's to clear.
- One target per call; re-run for multiple paths (each gets its own timestamped graveyard
  entry, so provenance stays per-artifact).
