---
id: zn5hckwgbzefzdotvnvpcyr
title: Generate_datasets_table
desc: ''
updated: 1783476228725
created: 1783476228725
---

## 2026.07.07 - Supported-datasets table generator

Generates the paper's supported-datasets table so it is never hand-transcribed
again. The script is deliberately thin: it owns only the dataset registry (the
source of truth) plus the surrounding prose, and delegates all signal
computation, caching, and md/LaTeX rendering to `[[torchcell.paper.tables]]` --
so the next paper table follows the same pattern instead of copy-pasting logic.

Each registry row maps to exactly ONE built LMDB, which is why the old note's
compacted `smf/dmf/tmf` rows are split one-per-dataset here: every emitted row
then carries a real, per-dataset gzip signal instead of a shared "pending".

Emits two artifacts from the one registry: `sections/datasets_table.tex`
(`\input` into the SI) and the readable mirror
`[[paper.supported-datasets-and-databases]]`.

### Notes

- Run from repo root: `python paper/nature-biotech/scripts/generate_datasets_table.py`
  (all; slow only on Costanzo dmf/dmi). `--max-gb N` skips the giant LMDBs for a
  quick pass; `--only <substr>` recomputes one; results memoize in
  `dataset_signals_cache.json`.
- Computing from data surfaced stale curated counts to fix later: Kuzmin 2020 dmf
  is really n=632,797 (note said 256,862 -- old table admitted "approximate"),
  Kemmeren n=1,450 (note 1,484), and the Zelezniak metabolome is in fact built
  (n=95) though the note said "not yet ingested".
- Ohya's queryable LMDB lives under `$DATA_ROOT/database/data/torchcell/scmd_ohya2005`,
  not the ML `data/torchcell/` root -- the registry pins that explicitly.
