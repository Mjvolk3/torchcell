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

## 2026.09.30 - Findings retired (issue #525)

All five Phase 13 Findings are retired. Now asserted: an enriched manifest without DOI or title (either alone) is `provenance_complete=False`; `thesis.txt` and a nested `si/extra/notes.md` carry no `mineru-ocr` source while `paper.md` and `si/si1.md` do; `_bib`, an empty directory and one holding only empty subdirectories get no manifest and are absent from the report; a corrupt existing manifest raises `CorruptManifestError` with its exact path and is left in place; a skipped key reports its manifest's values; two items sharing `dupKey2020` raise `DuplicateCitationKeyError` with `(dupKey2020: A, B)`.

## 2026.09.30 - SI sidecar roles pinned (issue #564)

New `test_role_for_mineru_sidecars_under_si_are_ocr_roles` (sidecars under `si/` are `ocr_layout` / `ocr_image`; `si/si_data/images/plate.png`, `si/si_data/run_middle.json` and `si/Table_S2.json` stay `si_data`) and `test_captured_key_full_role_table` (the full `(path, role, source)` table of an `avsecEffectiveGeneExpression2021`-shaped key from `build_manifest`).

## 2026.09.30 - Sidecar boundary pinned

`test_role_for_mineru_sidecars_under_si_are_ocr_roles` now also asserts `si/Figure_S1_images/a.png`, `si/Table_middle.json` and `si/Table_content_list.json` stay `si_data`.

## 2026.10.01 - Collection names read through everything (issue #563)

`test_enriched_manifest_carries_zotero_metadata_and_attachment_sources` now expects a trailing `("everything",)` call: `capture._collection_names` reads `ZoteroLibrary.list_collections`, which pages every collection since [[torchcell.literature.zotero]] fixed #563.
