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

## 2026.10.01 - Per-PDF figure directories (issue #579)

Previous behavior: `main()` ran `shutil.rmtree(<out-dir>/images)` and then `copytree`, and every `si/si*.pdf` of a key shares the out-dir `si/`, so each SI PDF's OCR deleted the previous PDF's figures. On the live mirror (read-only check) `leeMappingCellularResponse2014` is missing 39 of 43 referenced SI figures and `ohyaHighdimensionalLargescalePhenotyping2005` 47 of 47.

Fix: a new `--images-dir` argument (relative to `--out-dir`, default `images`) names the PDF's own figures directory; `ocr.images_dir_for` passes `images` for `paper.pdf` (layout unchanged) and `images/<stem>` for every other PDF. The markdown and `<stem>_content_list.json` references `images/<file>` are rewritten to `<images-dir>/<file>`; a reference to a figure MinerU did not write exits 5 before anything is written. Figures are staged into a fresh directory inside the run's scratch, the PDF's previous directory is moved aside, the staged one renamed into place, and only the moved-aside copy (that PDF's own old figures) is removed; no sibling directory is touched. The runner prints `MINERU_VERSION=<mineru.version.__version__>` and `MINERU_DPI=<effective dpi>` (the `--dpi` value, else the default of MinerU's `load_images_from_pdf`) for `ocr.py`'s processing record.

Evidence: [[tests.torchcell.literature.test_run_mineru]] (`test_two_si_pdfs_both_keep_their_figures`, `test_a_rerun_replaces_only_its_own_figures`, `test_the_paper_keeps_the_flat_images_directory`, `test_a_reference_to_an_unwritten_figure_exits_5_and_writes_nothing`).
