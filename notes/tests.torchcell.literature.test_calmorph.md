---
id: 1j08ap5dx01za0nanspwa3l
title: Test_calmorph
desc: ''
updated: 1791270250069
created: 1791270250069
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): `extract_calmorph_parameters` with `pdf_kind` and `pdf_text` replaced at their import site: the scanned refusal (text layer never read), the born-digital path reads the layout text once, and the no-rows refusal, each with its exact message.

### Audit 2 notes applied

- Audit 2: the born-digital extraction test now compares against a literal dict, not `parse_calmorph_table` itself.
