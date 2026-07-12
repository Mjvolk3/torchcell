---
id: kych3pbc0uxk6132zymzmap
title: Drain_merge_queue
desc: ''
updated: 1783563685580
created: 1783563685580
---

## 2026.07.08 - Model-free deterministic drainer for serialized landings

This is the deterministic (no-model) worker that consumes the single-writer merge queue ([[scripts.merge_queue]]) and is the ONLY automated writer to `main`. It exists because the landing work -- rebase onto origin/main, ff-push the tip, close the PR, remove the worktree -- is entirely deterministic; running it as a model loop would burn tokens doing nothing on idle ticks. Serializing landings through this single drainer is what stops parallel worktrees (which share one `.git`) from corrupting each other's landings.

- Two triggers, one behavior: `/enqueue-merge` runs it the instant work arrives, and a `*/2` cron safety-net picks up branches orphaned by a session that died mid-drain.
- Self-guards with a non-blocking flock -- at most one drainer runs at a time; a trigger that loses the lock is fine because the active holder drains the whole queue.
- Lands worktree -> origin (`push HEAD:main`), never via a possibly-stale local `main`, with a fetch+rebase+push retry that rides out the semantic-release bump commit CI pushes after each landing.
- Each pass first SWEEPS free notes (weekly/standalone notes edited on main) to origin -- divergence-aware and isolated from landings, mirroring iBioFoundry-AI's drainer.
