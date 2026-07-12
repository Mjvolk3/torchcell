---
id: ka2fc2hcss6ok33lypiglhs
title: Merge_queue
desc: ''
updated: 1783563692629
created: 1783563692629
---

## 2026.07.08 - Serialize landings so parallel worktrees never corrupt each other

This is the SQLite-backed queue library behind the single-writer merge queue (a faithful replica of iBioFoundry-AI's design). It exists because all worktrees share one `.git` object store + stash list, and pre-commit stashes/rolls back -- so two landings finishing at once race the shared ref and corrupt each other. Making landings pass through an ordered queue serializes them by construction: one writer lands at a time, and worktree branches reach `main` via rebase + ff-only without clobbering. The deterministic drainer that consumes this queue is [[scripts.drain_merge_queue]].

- A queue (editable, reorderable `position` column), not a blocking mutex -- we want a backlog we can reorder/drop/clear, not just exclusion.
- Stdlib-only + NFS-safe pragmas (`journal_mode=DELETE`, `BEGIN IMMEDIATE` atomic claim) so it runs from any worktree, a cron shell, or a bare container.
- The CLI is the interface every actor shares: sessions `add`, the drainer `claim`/`done`/`block`, the enqueue skill `watch`/`banner` for its close signal.
- `foreign` is the enqueue-time tangle guard -- strips sibling `plan:` note commits a branch inherited from a stale local main before they land silently.
