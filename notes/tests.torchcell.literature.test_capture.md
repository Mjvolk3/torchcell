---
id: l5aluqueu2sx5frqsi2d3on
title: Test_capture
desc: ''
updated: 1791270240516
created: 1791270240516
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): `capture_by_doi` with a real `ZoteroLibrary` over the read-only `FakeZot` (no write methods exist), `fetch_si_data` and `ocr_artifact` replaced at their import site. The main PDF ("Full Text PDF") becomes `paper.pdf` though listed after the SI; the manifest is asserted whole (minus `created_at`): sources `zotero:attachment:<key>`, the SI URL, `mineru-ocr` for the OCR markdown, the Zotero md5, roles and sha256 recomputed with hashlib; an unknown collection key is kept as the key. Also: no SI and no OCR calls when not requested, extra URLs alone trigger SI fetching, and the DOI-not-found refusal leaves no mirror directory.
