---
id: hcn4gtr6ihh1vsk2fsdgfgr
title: Bacterial_generation_profile
desc: ''
updated: 1792098597932
created: 1792098597932
---

## 2026.10.15 - Reading the bacterial tail's generation cost off one array

`experiments/database/scripts/bacterial_generation_profile.py` folds one per-adapter
generation array into one row per adapter and projects the full 170-task array from it.
It runs no generation: it reads the `DONE` markers, each task's
`telemetry/resource_samples.csv` and each task's generate log, and reuses
`fold_samples` and `read_recycles` from
[[experiments.database.scripts.kg_build_adapter_resources]] rather than reimplementing
the fold.

Three choices in it are deliberate:

- **The membership is recomputed, never listed.** `tail_slots` takes the adapter the
  measured prefix ended on (`--tail-from`, default `EnvChemgenShiver2016Adapter`) and
  returns every slot of `build_adapter_map` after it. The prose lists of the bacterial
  tail in the 2026-10-15 lane reports name 45 adapters by author and are one short, so a
  list in a note is not a membership.
- **The projection charges every task a measured number and says which.** The preference
  order is this run at the task's real tier, another run at that tier, the serial run's
  per-adapter fold, and then `SAMPLER_INTERVAL_S` for an adapter the serial run never
  sampled, which bounds it above rather than estimating it. The `basis` string of each
  tier row carries the counts, so a reader can see how much of a tier is a real
  measurement. Nothing is extrapolated past the largest thing observed.
- **A tier proposal is the smallest tier that holds the peak under the threshold**
  (`_next_tier_holding`), not the next tier up, and an adapter that no tier holds is said
  to need one of its own instead of being silently put in the largest.

Outputs `experiments/database/results/bacterial_generation_profile_<label>.json` and,
with `--table`, every Markdown table of
[[experiments.database.bacterial-generation-profile]]. `--list` prints the membership
with no array to read. The companion
`experiments/database/scripts/bacterial_generation_tiers.yaml` is the committed tier
table with the standard tier's concurrency at 3 instead of 4, handed to the conductor as
`TIERS=` so a measurement reads the committed table rather than editing it.
