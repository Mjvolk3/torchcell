---
id: g09wlo2p9ai9lgxhhee1c7s
title: Dango_string_version_sweep
desc: ''
updated: 1788466936842
created: 1788466936842
---

## 2026.09.03 - DANGO replication by STRING release, pulled from wandb

Script: `experiments/005-kuzmin2018-tmi/scripts/dango_string_version_sweep.py`. The results note
[[experiments.005-kuzmin2018-tmi.results]] recorded the STRING 9.1 / 11.0 / 12.0 sweep by reading
values off wandb charts. This script pulls every run of `zhao-group/torchcell_005-kuzmin2018-tmi_dango`
through the API, keeps runs with at least 100 logged epochs (drops the one 9-epoch smoke test), and
records per run the maximum over epochs of `val/gene_interaction/Pearson` (the checkpoint-selection
rule; an upward-biased order statistic, so epochs logged sit next to it). Outputs:

- `experiments/005-kuzmin2018-tmi/results/dango_string_version_sweep.csv` (19 runs) and
  `dango_string_version_summary.csv` (mean, SD, SEM per release x schedule)
- `paper/nature-biotech/sections/tab-dango-string-versions.tex`
- the panel below; `--from-csv` re-renders offline from the frozen run table.

Measured: best validation Pearson across the 19 runs spans 0.415 to 0.427; per release x schedule
means are 0.419 to 0.424 with SEM at most 0.003 where n > 1. No release or schedule separates from
the others beyond the run-to-run spread. These are validation maxima, not the test-split baseline of
the main text.

![](./assets/images/005-kuzmin2018-tmi/dango_string_version_sweep.svg)
