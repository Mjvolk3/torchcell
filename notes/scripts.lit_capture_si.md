---
id: v73jba035dy1uxwhj4zy402
title: Lit_capture_si
desc: ''
updated: 1791364449293
created: 1791364449293
---

## 2026.10.07 - Thin wrapper

Loads the repo `.env` and calls `capture_si.main`; the routes, outcomes and manual path are in [[torchcell.literature.capture_si]]. Sources: citation keys (positional), `--collection <name>` (group library, read only, repeatable) and `--doi` (a DOI already in a mirror manifest). `--dry-run` resolves and lists without downloading. `--ocr` runs MinerU on every SI PDF captured this run and records the outputs; it needs a GPU, so it is off by default. `--mirror-root` points at another `torchcell-library` directory and `--report-dir` moves the JSON report (default `<mirror-root>/_sync_reports/si_<UTC stamp>.json`, `si_dryrun_` for a dry run). Exit 1 when any key `failed`.
