---
id: u6x3hb8te68pwoxzhcps0ww
title: Extract
desc: ''
updated: 1783563720853
created: 1783563720853
---

## 2026.07.08 - Prefer the exact text layer over OCR, and decide which regime a PDF is in

This module exists because OCR is the fallback, not the default: a born-digital PDF already carries its data as a recoverable text layer that is exact, deterministic, and instant -- strictly better than OCR for tables, where a VLM pass can silently drop rows. The core job is to classify a PDF as `born_digital` vs `scanned` so the pipeline routes it correctly: trust the text layer (`pdftotext -layout`) when it is real, and only escalate to the high-DPI VLM OCR of [[torchcell.literature.ocr]] / [[torchcell.literature.scanned]] when it is absent or untrustworthy (including Acrobat-OCR'd scans, whose text layer must NOT be trusted).

- `pdf_kind`: born-digital requires real embedded fonts, a substantive text layer, and no page-sized image on most pages -- the last condition is what catches OCR'd scans masquerading as text PDFs.
- `pdf_text(layout=True)` preserves column geometry, which is what makes header-anchored parsing of born-digital tables (e.g. [[torchcell.literature.calmorph]]) reliable.
- Built entirely on the poppler CLIs (`pdffonts`/`pdfimages`/`pdftotext`/`pdfinfo`) -- a ubiquitous system dependency, no Python PDF library required.
