---
id: ar45jfvcxxsjlxcsxei7s6c
title: Test_ocr
desc: ''
updated: 1790780619550
created: 1790780619550
---

## 2026.09.30 - Phase 18: the MinerU runner on a recorded subprocess

New file, sixteen functions (19 cases), 20 to 100 percent. The exact default command, env and kwargs (`capture_output`, `text`, `timeout=3600`) with the parent `os.environ` unchanged; arguments beating the environment (`dpi=0` and `device_mode`) and environment defaults otherwise; the `_hf_home` order across four cases; refusals (a nonzero exit keeping the last 2000 characters of stderr, exit 0 with no markdown, a non-integer DPI variable before any subprocess, `TimeoutExpired` unwrapped); the two exact log lines; `ocr_artifact` running the paper then the SI files, the names the glob skips, kwargs forwarded, stopping at the first failure.

Findings: no existence check on the PDF (lines 82-110); no `mineru_version`, arguments or DPI are recorded, against the provenance rule in CLAUDE.md, and the command carries no version pin; SI files sort lexicographically so `si10` runs before `si2` (135); a directory with no `paper.pdf` returns SI-only results or `[]` without refusing; `mormino2022.py` line 434 names `torchcell.literature.ocr.run_mineru`, which does not exist.

## 2026.10.01 - Findings retired (issue #546)

All five findings are retired. Now asserted: a missing PDF or a directory path raises `PdfNotFoundError` with the exact message and no subprocess call; each OCR writes `<stem>_ocr_provenance.json` whose whole JSON is compared (version, effective and requested DPI, arguments, device, `images_dir`, exact command, PDF sha256); a runner that misreports its facts raises `RunnerReportError` with the exact message (three parametrized cases); `ocr_artifact` runs `si1, si2, si10` in natural order with per-PDF `--images-dir`; a directory without `paper.pdf` raises `MissingPaperPdfError`; Mormino's processor resolves to `ocr.ocr_pdf`. The command now ends with `--images-dir images` (paper) or `--images-dir images/<stem>`.
