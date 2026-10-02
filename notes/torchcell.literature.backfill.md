---
id: xl7m1m382uf1z1wbi75a32q
title: Backfill
desc: ''
updated: 1790814989907
created: 1790814989907
---

## 2026.09.30 - Completeness, key directories, corrupt manifests, duplicate keys

Issue #525 (Phase 13 Findings, PR #522). Previous behavior: an enriched manifest was always `provenance_complete=True`, even with no DOI or title; every subdirectory of the mirror (including `_bib` and empty directories) got a manifest; an existing `manifest.json` was skipped without being read, so a corrupt one survived; two Zotero items sharing a citation key collapsed to the later one silently.

Fix:

- `provenance_complete` is `doi is not None and title is not None` for an enriched manifest, matching the `_METADATA_FIELDS` comment.
- `is_citation_key_dir`: a directory whose name does not start with `_` and that holds at least one file. `backfill_mirror` visits only those; the rest get no manifest and are absent from the report.
- Without `force`, an existing manifest is parsed; a parse failure raises `CorruptManifestError` naming the path. A skipped result now reports the existing manifest's file count, completeness and null metadata.
- `build_citation_index` raises `DuplicateCitationKeyError` listing every shared key with all of its item keys.

Evidence: [[tests.torchcell.literature.test_backfill]], tests `test_enriched_manifest_without_doi_or_title_is_incomplete`, `test_enriched_completeness_needs_both_doi_and_title`, `test_citation_index_refuses_items_sharing_a_citation_key`, `test_existing_corrupt_manifest_raises_with_its_path`, `test_skipped_key_reports_what_its_existing_manifest_records`, `test_mirror_scan_visits_only_citation_key_directories`.

## 2026.10.02 - Citation index over top-level items only (issues #607, #644)

Previous behavior: `build_citation_index` scanned every Zotero item (`everything(items())`), so attachments, notes and annotations, which carry no citation key, fell through `_resolve_citation_key` to a generated key and collided as `unknownFullTextPDFXXXX`, `unknownSnapshotXXXX` and similar. The SI re-OCR (slurm 3241) refused with a `DuplicateCitationKeyError` listing more than a hundred such phantom duplicates beside the one real duplicate, `chaoPredictingDynamicExpression2025` (items 8WK5B5MP, Y6J4Q887, issue #644).

Fix:

- `citation_index_with_duplicates(lib)` reads `everything(top())` (the `/items/top` endpoint, paginated) and skips every item whose `itemType` is in `CHILD_ITEM_TYPES` (`attachment`, `note`, `annotation`), since a standalone attachment or note is top-level. It returns `(index, duplicates)`; a duplicated key is absent from the index.
- `build_citation_index` wraps it and still raises `DuplicateCitationKeyError` on any duplicate, so `backfill_mirror` keeps refusing (a mirror directory can be any key). `describe_duplicates` formats the list for both callers.
- Attachments are still reached from their parent through `ZoteroLibrary.pdf_attachments` in `_enriched_manifest`, never through the index.

Evidence: [[tests.torchcell.literature.test_backfill]], tests `test_citation_index_reads_top_level_items_only`, `test_citation_index_skips_a_standalone_child_type` (one per type), `test_citation_index_with_duplicates_leaves_a_shared_key_out_of_the_index`, `test_backfill_mirror_refuses_any_duplicate_before_writing`.
