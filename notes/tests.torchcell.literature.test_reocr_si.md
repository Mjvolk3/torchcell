---
id: kfytb417dws5x198feh3k0u
title: Test_reocr_si
desc: ''
updated: 1790882708462
created: 1790882708462
---

## 2026.10.01 - Order, refusals and retirement with OCR and Zotero faked

`ocr_pdf` is a fake that writes the per-PDF figure and markdown the runner writes; Zotero is three recorders; `retire_flat_figures` runs the real `scripts/deprecate.sh` into a `tmp_path` graveyard. Asserted exactly: natural SI order and the no-SI refusal; flat figures are only files directly in `si/images/`; unresolved references by markdown name (markdown and HTML forms, prose URL ignored); the full event order across two keys, the graveyard entry name, its contents and its `DEPRECATION.txt` lines; an OCR failure retires nothing and writes no manifest; an unresolved reference refuses with the exact message before retiring; `main` arguments.

## 2026.10.01 - Delta review: exact phase sequence and refusals

Check and retire are wrapped to record each call, so the exact event sequence for two keys with flat figures is asserted (Zotero, OCR x4, check x2, retire x2, check x2, backfill x2). Added: an OCR failure on the second key retires nothing; a key missing from Zotero refuses after only the index call; a non-enriched backfill refuses; a dangling flat reference refuses after retirement; a graveyard inside `DATA_ROOT` is refused by the real `deprecate.sh` (exit 2) with the figures moved back; a leftover staging directory is retired; `main` passes `--graveyard`, `--device-mode` and `--root` through the real `reocr_keys`.

## 2026.10.01 - Moved from tests/scripts and hardened (second delta review)

Moved from `tests/scripts/test_lit_reocr_si.py` (CI runs only `tests/torchcell`) with the logic now in [[torchcell.literature.reocr_si]]; the sections above came from the retired note `tests.scripts.test_lit_reocr_si`. An autouse fixture points `DEFAULT_GRAVEYARD` and `DATA_ROOT` into `tmp_path`, so no test or mutant can reach `/scratch/projects/torchcell-deprecated`. The fake `backfill_key` now writes `manifest.json` unless `dry_run`, so the not-enriched refusals are asserted against the bytes on disk (dry-run refusal before any OCR; final refusal restores the previous bytes; no previous manifest leaves none). Added: the up-front refusal of markdown with no SI PDF that references a flat figure; the after-retire backstop driven by an OCR that writes a flat reference; a `deprecate.sh` that moves then exits 3 (exit code kept, graveyard named); a staging directory holding every figure with nothing flat (a kill inside `deprecate.sh`). Twelve mutants of the module each fail at least one test.
