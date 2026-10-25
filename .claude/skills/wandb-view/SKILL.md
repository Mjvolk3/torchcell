---
name: wandb-view
description: Answer "show me the wandb view / group / runs / plots for round X" with the loss view FIRST, then the hardware table and round-level links, one URL per line; every run link carries the round view (?nw=) so its loss plots are already built; the saved view holds a panel for EVERY logged key, loss first; no reports as the deliverable.
---

# W&B View

Use this whenever the user asks for a W&B view, group, workspace, "the runs", "the
charts", or "the plots" for a round or a run. The review unit is the round and its arms;
a run link is given only as the example of a named phenomenon, and then it must open
with the plots already built.

## Rules that come from the owner, in order of how often they were broken

1. **The first link in the answer is the loss view** (train/loss and val/loss on one
   axis, per arm). Owner, 2026-10-24: "when you give me wandb first thing i want to see
   is train and loss curves". Jobs, tables and everything else come after that link.
2. **Every run link carries the round view**: write
   `https://wandb.ai/zhao-group/<project>/runs/<id>?nw=<round_view_id>`, never a bare
   `/runs/<id>`. A bare run page opens the owner's personal workspace, which the API
   cannot write, so the loss plots are not there and the owner has to build them. Owner,
   2026-10-24: "on the links you need to create the loss plots for easy viewing i never
   want to have to create them". The `nw` parameter applies the saved view's panel layout
   to the run page.
3. **The saved view carries a panel for EVERY key the round's runs log**, not a curated
   subset. Owner, 2026-10-24: "i don't like reports... i just like to see the plots from
   specific runs but you need to populate all of the plots... i like to scroll through to
   see what is being logged." Build the key list from the union of the runs' summaries at
   run time, so a new key appears on the next rerun; verify through the API that no
   logged numeric key lacks a panel. A W&B Report may exist for one-arm-only panels (a
   saved view has one runset), but it is never the first or the only link.
4. **Train and validation loss together** on one axis, first panel, every view (memory
   `always-show-train-and-validation-loss-together`); then each alone; then the
   per-dataset train+val pairs, one panel each.
5. **One URL per line, nothing else on the line** (tmux, no wrap, click).

## The answer, in this order

**1. The loss view link** for the round (rule 1).

**2. Hardware table.** One row per round the question touches. Columns are fixed:

| round | cluster / partition | card (runs per card) | job |
|---|---|---|---|
| v13 expression splits | IGB `gpu` | A40 48 GB (4 per card) | 2397311 |
| 043 sweep 2026-10-24 | GilaHyper `main` | RTX 6000 Ada 48 GB (1 per card) | 6337-6377 |
| 043 Aurora 2026-10-24 | Aurora `capacity`, 1 node | Data Center GPU Max 1550 (1 per tile) | PBS 8959835, 8959861 |

Get the facts from the launcher (`sbatch -p`, `--gres`, `RUNS_PER_GPU`) and from
`squeue -j <id> -o "%i %A %P %N"` or `sacct -j <id> -o JobID,Partition,NodeList,AllocTRES`
on the login node (a query, not compute). Card types seen so far: IGB `gpu` = A40 48 GB,
`cabbi` (compute-3-3) = RTX 6000 Ada 48 GB, GilaHyper `main` = RTX 6000 Ada 48 GB,
Polaris = A100 40 GB (4 per node), Aurora = Data Center GPU Max 1550 (12 tiles per node);
`mmli` (compute-5-7) unrecorded, read it from `sinfo -p mmli -o "%N %G"` before stating
it. Array tasks can carry non-contiguous job ids; list the array id in the table and the
task ids only when they matter.

**3. Round-level links, each URL alone on its own line:**

- the round view (every run, lines grouped by arm): `https://wandb.ai/zhao-group/<project>?nw=<view_id>`
- per arm, TWO lines: the populated arm view `https://wandb.ai/zhao-group/<project>?nw=<arm_view_id>`
  (only that arm's runs, one line per run), then the group workspace
  `https://wandb.ai/zhao-group/<project>/groups/<arm>/workspace`. Always give the
  `/workspace` form of a group link, never the bare `/groups/<arm>`.

The group workspace is a personal view and the API refuses to write it ("does not
currently support user views"), so the arm view is where the panels live. Say that in one
sentence when giving both links; do not claim the group page itself was populated.

**4. Example runs, only for a named phenomenon** (a collapse, an early peak, a
divergence), as `<id> <arm> <what it shows>` followed by its URL on the next line, with
the `?nw=<round_view_id>` suffix (rule 2).

**A phenomenon named in chat gets a panel in the saved view first.** Before citing runs
for something seen in the curves, make sure the view has the panel that shows it (overlay
the metrics on one panel when the point is a contrast, e.g. train against validation),
rerun the script, verify through the API, and give the view link.

## Keeping the view populated

The personal default workspace cannot be written by the API ("does not currently
support user views"), so every round has a SAVED view maintained by a committed script.
Two scripts exist, one per experiment family:

```bash
~/miniconda3/envs/torchcell/bin/python experiments/019-simb-multimodal/scripts/wandb_v13_report.py --round v14
~/miniconda3/envs/torchcell/bin/python experiments/043-expression-proteome-morphology/scripts/wandb_view_043.py --round ghyper_sweep_2026_10_24
```

Each script holds one `Round` entry per round in `ROUNDS` (run ids, arms, view ids),
renames runs `<arm>_seed<k>`, sets the W&B group to the arm family, overwrites the saved
round view and every arm view in place, and records new view ids in the experiment's
`results/wandb_view_ids.json` (committed). The 043 script builds its section list from
the union of the runs' logged keys: 1 Loss (train+val on one axis, each alone, the
per-dataset train+val pairs), 2 Validation metrics by dataset, 3 Train-eval metrics
(overlaid with val where the metric matches), 4 Doubles, promoter, protein, expression
aggregates, 5 Optimization (grad norm log y, clip fraction, lr, step loss, operator
norms), 6 Time and memory, 7 Setup and bookkeeping; x = epoch for epoch-logged keys,
optimizer step for step-logged ones; once-written setup keys become bar panels; any key
no rule names lands in section 7. Round views group by arm (mean with min/max band), arm
views draw one line per run. A view keeps only keys its own runs log, so no panel is an
empty box. **A sync of a live run resets name and group, so rerun the script after every
sync**; it is idempotent. Verify through the API (panel counts per section, the plotted
key set equals the logged key set) before saying it is populated.

Offline IGB runs show `finished` once synced whatever their training state; take the
epoch from the run history, and call an in-flight round partial. Live ALCF runs can show
`crashed` from heartbeat gaps; confirm with the job log before calling a run dead.

## Related

- `.claude/skills/monitor-tcdb-build` for sync helpers on IGB
- memory `wandb-group-links-per-arm`, `wandb-charts-view-after-every-round`,
  `wandb-run-links-one-per-line`, `wandb-no-reports-views-with-every-plot`,
  `always-show-train-and-validation-loss-together`
