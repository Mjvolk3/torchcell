---
id: uix0tswms32eyoduc7ez25t
title: Test_lit_reocr_si
desc: ''
updated: 1790876242390
created: 1790876242390
---

## 2026.10.01 - Order, refusals and retirement with OCR and Zotero faked

`ocr_pdf` is a fake that writes the per-PDF figure and markdown the runner writes; Zotero is three recorders; `retire_flat_figures` runs the real `scripts/deprecate.sh` into a `tmp_path` graveyard. Asserted exactly: natural SI order and the no-SI refusal; flat figures are only files directly in `si/images/`; unresolved references by markdown name (markdown and HTML forms, prose URL ignored); the full event order across two keys, the graveyard entry name, its contents and its `DEPRECATION.txt` lines; an OCR failure retires nothing and writes no manifest; an unresolved reference refuses with the exact message before retiring; `main` arguments.
