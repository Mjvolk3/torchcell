---
name: drain-merge-queue
description: Run the deterministic merge-queue drainer once (land queued worktree branches to main). Normally triggered automatically by /enqueue-merge and a cron; use this only to force a drain by hand.
---

# Drain Merge Queue

Run the deterministic, **model-free** drainer (`scripts/drain_merge_queue.py`)
one time:

```bash
$HOME/miniconda3/envs/torchcell/bin/python \
  $HOME/Documents/projects/torchcell/scripts/drain_merge_queue.py
```

That script self-guards with a non-blocking `flock` (at most one drainer runs at
a time), then claims and lands every queued branch **worktree -> origin**
(`rebase origin/main` -> `push HEAD:main` -> close PR -> clean up worktree +
branch). It is the **only** automated writer to `main`, and it never touches
local `main`, so a diverged local `main` cannot block a landing. It also sweeps
free notes edited on `main` (weekly task notes and other standalone notes -- not
paired module/experiment/script notes) to origin at the start of each pass,
mirroring iBioFoundry-AI. It prints a one-line summary: what swept, what landed,
what blocked.

## You normally never call this by hand

The drainer is **event-driven**, not a poll:

- `/enqueue-merge` runs it right after adding a branch, so a landing fires the
  instant work arrives.
- A `*/2 * * * *` crontab entry runs it as a cheap, deterministic safety-net: it
  picks up any branch orphaned by a session that died mid-drain (the OS releases
  that session's `flock` on death, but its un-landed rows still need a drainer).

Both cost **zero tokens** -- the drainer is pure git/gh orchestration.

## When to use this skill

Only to **force a drain now** -- e.g. right after `merge_queue.py requeue`-ing a
branch you just fixed, or to flush the queue without waiting up to 2 min for the
cron. If a branch comes back `blocked`, fix it in its worktree and `requeue`.
