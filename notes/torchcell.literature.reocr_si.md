---
id: vcur6n7svie8hka3lufmnov
title: Reocr_si
desc: ''
updated: 1790882700628
created: 1790882700628
---

## 2026.10.01 - Re-OCR repair logic (issue #579, PR #585)

The logic of `scripts/lit_reocr_si.py` lives here so CI (`pytest tests/torchcell`) runs its tests and diff-cover sees it; the script is a thin wrapper that loads `.env` and calls `main`. Phases over every key: precheck (Zotero index, dry-run `backfill_key` must be `enriched`, no markdown without an SI PDF referencing a flat figure), OCR, reference check, retirement through `scripts/deprecate.sh` with the full environment (its `DATA_ROOT` guard applies; on failure the figures are moved back, or the error names the graveyard when `deprecate.sh` had already moved them), backstop check, backfill with the previous manifest bytes restored unless `enriched`. An item with no DOI or title still backfills as `enriched` with `provenance_complete=False`; the check passes it. Tests: [[tests.torchcell.literature.test_reocr_si]].

## 2026.10.02 - Precheck refuses only a duplicate of a processed key (issue #607)

`reocr_keys` now calls `citation_index_with_duplicates` from [[torchcell.literature.backfill]] and decides explicitly: a duplicated key that is among the requested keys raises `DuplicateCitationKeyError` ("Zotero items share a requested citation key, nothing OCR'd (...)") naming only the requested ones; any other duplicate is logged once at WARNING with the full list and the run proceeds. Before, any duplicate anywhere in the library blocked the run (slurm 3241, blocked by `chaoPredictingDynamicExpression2025`, issue #644, plus phantom child-item keys). Tests: `test_a_duplicate_of_a_key_not_processed_warns_once_and_the_run_proceeds`, `test_a_duplicate_of_a_processed_key_refuses_by_name_before_any_ocr` in [[tests.torchcell.literature.test_reocr_si]].
