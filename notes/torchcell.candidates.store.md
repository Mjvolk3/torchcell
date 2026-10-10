---
id: vxr2kl00o4tsnlc3oqz1jqb
title: Store
desc: ''
updated: 1791617785736
created: 1791617785736
---

## 2026.10.10 - The tracked verdict store and the grandfather list

Verdicts live at `database/candidates/<citation_key>.json`: tracked so CI reads them, beside `database/releases/`, outside the supported-queries hook pattern, and outside the wheel (plan decision 4). `write_verdict` writes `model_dump_json(indent=2)` plus a newline; `read_verdict` refuses a file whose verdict names another key; `verdicts_by_row` refuses two verdicts for one row.

`GRANDFATHERED` is every class in `dataset_registry` after importing every module under `torchcell/datasets` on 2026-10-10: 125 classes, sorted, frozen. It may only shrink. `enforcement_violations(registry, grandfathered, verdicts)` names every other class whose module lacks a `CITATION_KEY` or whose verdict is missing or not passing; `tests/torchcell/datasets/test_candidate_verdicts.py` asserts it is empty. `CITATION_KEY` is read off the defining module (measured: 87 of the 125 modules declare it at module level and no class declares it as a class attribute).

The store ships empty in this PR: a verdict records the commit it read, and a verdict written on a branch would name a commit the rebase replaces. The re-audit (piece 3) writes the first verdicts from main.
