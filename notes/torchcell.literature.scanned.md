---
id: t17o8rbzcoa5a9cd38irliw
title: Scanned
desc: ''
updated: 1783563763159
created: 1783563763159
---

## 2026.07.08 - Self-verifying multi-pass OCR so a scan's completeness is proven, not hoped

This module exists because a scan has no exact text layer to fall back on, so OCR itself must carry completeness -- and a single VLM pass provably drops rows from dense tables. It exploits a key fact: because layout detection runs on the page image at the chosen DPI, DIFFERENT DPIs drop DIFFERENT rows. So it OCRs across a DPI sweep, accumulates the union of everything found (recall climbs monotonically), and runs a shape-of-data oracle after each pass, stopping the moment coverage is provably complete. This is what lets us claim a scanned table was captured in full rather than trusting one lucky pass.

- `shape_check`: judges completeness by data shape -- either an exact known-schema id set (reports precisely which rows are missing) or a caption-advertised row count.
- `extract_scanned`: drives the sweep via [[torchcell.literature.ocr]], escalating DPI only until the oracle clears; effective resolution saturates near ~360 DPI (Qwen's pixel budget), so cranking to 600 is wasted.
- Deliberately scoped to row COVERAGE, not cell-value correctness -- value precision on noisy scans is a separate cross-pass-agreement step.
