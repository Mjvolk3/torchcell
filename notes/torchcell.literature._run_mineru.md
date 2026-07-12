---
id: jdedpi6gnydsumazesa9cx6
title: _run_mineru
desc: ''
updated: 1783563756115
created: 1783563756115
---

## 2026.07.08 - The subprocess entrypoint that keeps MinerU's toxic deps out of torchcell

This module exists as the process boundary that lets torchcell use MinerU without importing it: MinerU pins torch<2.11 + PaddleOCR (incompatible with the torchcell env), so the OCR work has to happen inside a separate conda env, and this standalone script is that env's entrypoint. [[torchcell.literature.ocr]] shells out to it; it imports only `mineru` + stdlib so it stays loadable in the minimal env, runs one PDF, and flattens MinerU's nested output so `<out-dir>/<stem>.md` lands next to its `images/`.

- Sets `HF_HOME` BEFORE importing MinerU (which reads the cache path at import time), falling back to `$DATA_ROOT/models/mineru/hf_cache`.
- Monkey-patches MinerU's page-rasterization DPI, which `do_parse` does not otherwise expose -- the mechanism behind the DPI knob that recovers dropped table rows.
- Communicates via explicit exit codes (2 PDF missing / 3 no markdown / 4 HF_HOME underivable) so the parent can fail loudly rather than silently skip. Adapted from Swanki's `run_mineru_swanki.py`.
