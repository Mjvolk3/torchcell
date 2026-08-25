---
id: t066jse4dne4yow1ho5fxo3
title: Wandb_sync_offline
desc: ''
updated: 1787697368463
created: 1787697368463
---

## 2026.08.25 - One-shot offline sync that keeps its own books

`scripts/wandb_sync_offline.sh` syncs offline W&B runs from a cluster login node.

```bash
bash scripts/wandb_sync_offline.sh                # every offline run
bash scripts/wandb_sync_offline.sh 2317356        # only paths matching a pattern (e.g. job id)
bash scripts/wandb_sync_offline.sh '' --recheck   # ignore our markers, re-attempt everything
```

Defaults: `WANDB_OFFLINE_ROOT=$HOME/scratch/torchcell/wandb-experiments`,
`WANDB_SYNC_LOG=$HOME/scratch/torchcell/wandb_sync.log`.

### Why it writes its own marker

`wandb sync` writes `run-<id>.wandb.synced` only **sometimes**. Verified on IGB 2026-07-28: a
run synced successfully (exit 0, "... done.") and still got no marker, even with `--mark-synced`
passed explicitly -- while `wandb sync --sync-all` on the same directory immediately afterwards
reported "Nothing to sync", i.e. wandb knew the run was synced but had not recorded it anywhere
observable.

Trusting that marker means those runs count as **pending forever**, every pass re-uploads them,
and the pending total never falls no matter how many passes run. That was the reported symptom:
~150 runs pending across 2,853 dirs, indefinitely. The DATA was always fine; the accounting was
not.

So the script writes its own `.tc-synced` (timestamp + resulting run URL) on a zero exit. The
pass is then **idempotent** and observably complete. Existing wandb markers were seeded across
1,529 dirs when this landed, so no already-uploaded run is re-sent.

### Run it ONCE, by hand, on the login node

Never loop it. A persistent background process on a shared login node is exactly what the
head-node policy forbids, and an unattended loop cannot be seen or reaped by the user. Compute
nodes have no internet, so the sync **must** run on the login node -- see the cluster rules in
[[user.Mjvolk3.torchcell.tasks.weekly.2026.31]] for the IGB/Delta split.

The path filter exists so a single job can be synced without walking every directory.
