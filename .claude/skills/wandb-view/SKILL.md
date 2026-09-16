---
name: wandb-view
description: Answer "show me the wandb view / group / runs for round X" with the hardware table plus round-level group and saved-view links, one URL per line, no per-run lists; maintain the saved Charts view with the committed report script.
---

# W&B View

Use this whenever the user asks for a W&B view, group, workspace, "the runs", or "the
charts" for a round. The review unit is the round and its arms, never a run.

## The answer has two parts, in this order

**1. Hardware table.** One row per round the question touches. Columns are fixed:

| round | cluster / partition | card (runs per card) | job |
|---|---|---|---|
| v13 expression splits | IGB `gpu` | A40 48 GB (4 per card) | 2397311 |
| v14 proteome | IGB `cabbi` compute-3-3 | RTX 6000 Ada 48 GB (4 per card) | 2400350 |
| baselines | GilaHyper CPU | none | 1976 |

Get the facts from the launcher (`sbatch -p`, `--gres`, `RUNS_PER_GPU`) and from
`squeue -j <id> -o "%i %A %P %N"` or `sacct -j <id> -o JobID,Partition,NodeList,AllocTRES`
on the login node (a query, not compute). Card types seen so far: IGB `gpu` = A40 48 GB,
`cabbi` (compute-3-3) = RTX 6000 Ada 48 GB; `mmli` (compute-5-7) unrecorded, read it from
`sinfo -p mmli -o "%N %G"` before stating it. Array tasks can carry non-contiguous
job ids (v14: 2400350, 2400351, 2401107, 2401108); list the array id in the table and
the task ids only when they matter.

**2. Round-level links, each URL alone on its own line** (tmux, no wrap, click):

- the saved Charts view: `https://wandb.ai/zhao-group/<project>?nw=<view_id>`
- one group page per arm: `https://wandb.ai/zhao-group/<project>/groups/<arm>`
- the report, if one exists

Do NOT list individual runs. An individual run link appears only as the example of a
named phenomenon (a collapse, an early peak), and then as `<id> <arm> <what it shows>`
followed by its URL on the next line.

**A phenomenon named in chat gets a panel in the saved view first.** Before citing runs
for something seen in the curves, add a section or panel to `chart_sections` that shows
it (overlay the metrics on one panel when the point is a contrast, e.g. train against
validation for the interpolation watch), rerun the script, verify through the API, and
give the view link. The user reads the charts, not the run pages.

## Keeping the view populated

The personal default workspace cannot be written by the API ("does not currently
support user views"), so every round has a SAVED view maintained by a committed script:

```bash
~/miniconda3/envs/torchcell/bin/python experiments/019-simb-multimodal/scripts/wandb_v13_report.py --round v14
```

The script (one `Round` entry per round in `ROUNDS`) renames runs `<arm>_seed<k>`, writes
config keys `arm/split/readout/partition` so the group page is the arm, overwrites the
saved view with ranked sections (1 headline validation, 2 train side, 3 masked
conditioning, 4 error and calibration, 5 optimization, 6 bookkeeping; x = epoch, grouped
by arm), and updates the report in place by title. Rerun it after every sync. A new round
starts with `view_id=None`; the first run calls `save_as_new_view()`, and the returned id
is pinned into `ROUNDS` and committed. Verify the view through the API (panel counts per
section, every metric key present in a run summary) before saying it is populated.

Offline IGB runs show `finished` once synced whatever their training state; take the
epoch from the run history, and call an in-flight round partial.

## Related

- `.claude/skills/monitor-tcdb-build` for sync helpers on IGB
- memory `wandb-group-links-per-arm`, `wandb-charts-view-after-every-round`,
  `wandb-run-links-one-per-line`
