---
id: li6ggwihzrypktrd8sxv15f
title: Test_extract
desc: ''
updated: 1791270217350
created: 1791270217350
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): the poppler wrappers with `extract._run` replaced by a recorder answering exact command tuples with hand-written `pdffonts`, `pdfinfo`, `pdftotext` and `pdfimages -list` output; `_run` itself checked against a stubbed `subprocess.run` (`capture_output=True, text=True, check=True`).

- Page-sized images: a 612 x 792 pt page at 300 ppi is 2550 x 3300 px, ratio 1.0; a 600 x 400 image at 150 ppi is 0.114; a zero-ppi smask, a short row and a zero-area page are skipped; the 0.5 threshold is inclusive (612 x 396 at 72 ppi counts, 612 x 395 does not).
- `pdf_kind`: no fonts, fewer than 20 words, a page-sized image on every page, and the born-digital case, each with the exact call sequence.

Finding: `pdf_kind` calls a PDF scanned only when EVERY page carries a page-sized image (`count >= max(1, pages)`), while its docstring and the module comment say MOST pages; a 4-page scan with one text-only cover page is classified born digital (extract.py:124).

### Audit 2 notes applied

- Reach (audit 2): the only caller of `pdf_kind` is calmorph on a born-digital SI, so latent. Added an anisotropic-ppi test (x 300 / y 150 and x 150 / y 300) so each page axis must use its own ppi.
