---
id: koccv1fpi0ei4ub6c70k6l0
title: Test_backfill
desc: ''
updated: 1790762198529
created: 1790762198529
---

## 2026.09.30 - Phase 13: every path on a synthetic mirror with a fake Zotero

Five to twenty tests, 72 to 100 percent. `FakeZot` stands in for the library with `ZoteroLibrary.from_env` patched and `load_dotenv` stubbed in `main`; whole `manifest.json` equality offline and enriched (sorted file order, attachment sources and md5s, collection names with an unresolved key falling back to itself), the Zotero read-call sequence, a matched item with no library written offline, force re-hashing changed bytes, force plus dry-run writing nothing, missing credentials raising `KeyError`, the report counters and three `main` summary lines.

Findings: an enriched manifest is `provenance_complete=True` even with no DOI or title (line 114, against the comment at 44); a top-level `thesis.txt` is tagged `mineru-ocr` (`manifest.py` line 261); every subdirectory counts as a citation key, so `_bib` and an empty directory each get a manifest (200); a corrupt existing manifest is skipped without being read (139-140); two items with the same citation key, the later wins silently (95).
