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
