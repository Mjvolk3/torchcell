---
id: ie3amiq7ezigu7lcxv5xfpr
title: Ocr
desc: ''
updated: 1783563734958
created: 1783563734958
---

## 2026.07.08 - A deterministic, isolated OCR recipe for the PDFs a text layer can't cover

This module exists to turn a paper's PDFs into markdown artifacts via a repeatable MinerU recipe -- the fallback path for scans and image-heavy pages that [[torchcell.literature.extract]]'s born-digital text layer cannot handle. The processing is provenance-worthy (tool + version + DPI recorded in [[torchcell.literature.manifest]]) so an OCR artifact is reproducible, not a one-off. MinerU pins an incompatible torch + PaddleOCR, so it is deliberately quarantined in its own conda env and invoked as a subprocess ([[torchcell.literature._run_mineru]]) -- never imported into the torchcell env.

- `ocr_pdf` / `ocr_artifact`: OCR one PDF (or all of paper.pdf + si/si*.pdf) to markdown alongside it, with no silent skips -- a non-zero MinerU exit or a missing output file raises.
- Exposes DPI as a first-class knob because rasterization resolution is the lever that recovers rows MinerU's default 200 DPI drops from dense tables; the multi-pass sweep that exploits this lives in [[torchcell.literature.scanned]].
- Resolves the HuggingFace model cache under `$DATA_ROOT` and the isolated MinerU interpreter via env override, so first-run model downloads are deterministic and relocatable.

## 2026.10.01 - Refusals, natural SI order and the OCR processing record (issues #546, #579)

Previous behavior: no existence check on the PDF (the runner's exit 2 was the only refusal), no MinerU version, arguments or DPI recorded, SI PDFs sorted lexicographically (`si10` before `si2`), and a directory with no `paper.pdf` returned SI-only results or `[]`.

Now: `ocr_pdf` raises `PdfNotFoundError` before starting MinerU; passes `--images-dir` from `images_dir_for` (`images` for `paper.pdf`, `images/<stem>` otherwise); and writes a `ProcessingRecord` (processor `torchcell.literature.ocr.ocr_pdf`, tool `mineru`, version and effective DPI as the runner printed them, `backend`, `lang`, `method`, `device_mode`, `dpi_requested`, `images_dir`, the exact `command`, and the PDF's sha256 as `input_sha256`) to `<stem>_ocr_provenance.json` beside the markdown. A runner that does not print `MINERU_VERSION` and `MINERU_DPI` exactly once raises `RunnerReportError`. `ocr_artifact` runs SI PDFs in natural order and raises `MissingPaperPdfError` when `paper.pdf` is absent; its only caller, `capture.capture_by_doi`, always downloads the article as `paper.pdf` first, so no caller relied on the SI-only result.

Evidence: [[tests.torchcell.literature.test_ocr]].
