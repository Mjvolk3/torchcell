---
id: r31lf5zwuomazkp16wwhex5
title: Coverage_gaps
desc: ''
updated: 1790409881272
created: 1790409881272
---

## 2026.09.26 - The campaign table generator

`scripts/coverage_gaps.py` turns `coverage json` files into the per-module markdown table the campaign note carries (Decisions 14, 17 and 20 of [[plan.test-suite-buildout.2026.09.25]]); a table in a note is pasted from its output, never typed. Columns: live-critical (the module is reached from a non-test root in the importer graph of [[scripts.legacy_partition]]), statement count, and one coverage column per run given (`--before`, `--after`, `--import-only`, `--local`), plus a delta when both before and after are present. Rows sort live-critical first, then ascending by the after column, and modules under 50 statements are hidden unless `--all`. The TOTAL rows give coverage.py's combined line+branch percent and the line-only percent from the same run; codecov reports the line figure, so the two are labeled.

`make cov-gaps` runs the behavioral suite (import-all deselected) and the import-only suite into two data files and prints the table. The CI test workflow runs the same two steps and uploads `coverage.json` / `coverage-import.json` as a workflow artifact, so the CI table can be regenerated locally. First table (local dev box, PR-0a tip, behavioral run only): TOTAL 23.3% line+branch, 25.4% line only, 68,290 statements; the live-critical modules at 0% are the `knowledge_graphs` build entry points, `sga/`, `ontology/`, `verification/runners.py`, `viz/graph_recovery.py` and `metabolism/parameters.py`.
