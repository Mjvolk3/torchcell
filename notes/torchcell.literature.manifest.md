---
id: s0siejlx1mf64fc6astuypo
title: Manifest
desc: ''
updated: 1783563727901
created: 1783563727901
---

## 2026.07.08 - The pydantic ground truth: per-file sha256 + provenance for one paper's mirror

This module exists to give every captured paper a single, serializable record of what it contains and where each byte came from -- written as `manifest.json` at the root of its artifact directory. It is the bottom of the import graph on purpose: the record types (`RetrievalMethod`, `RetrievalRecord`, `ProcessingRecord`, `ArtifactRecord`, `Manifest`) are pure pydantic data with no I/O, so [[torchcell.literature.provenance]] can layer verify/re-run BEHAVIOR on top without a circular import. The per-file sha256 is what lets any later run prove the mirror is intact, and the DOI is the join key back to Zotero and to the TorchCell dataset object.

- `build_manifest` scans an artifact directory, hashes and sizes every file, and tags each with a role (paper PDF, SI PDF, OCR markdown, SI data, MinerU byproducts) so the manifest is a complete inventory, not a curated subset.
- `ArtifactRecord` carries optional retrieval + processing sub-records so provenance survives serialize/reload -- one general per-file record serving papers, supplements, and dataset raw files alike.
- `si_expected` vs captured `si_data` gives a completeness check (what the paper says should exist vs what we actually mirrored); `si_data_sources` records the external repos so reproduction never needs the publisher.

## 2026.09.30 - mineru-ocr source by location

Issue #525. Previously `build_manifest` gave every `paper_ocr` or `si_ocr` file `source="mineru-ocr"`, so a top-level born-digital `thesis.txt` recorded an OCR step that never ran. Now `_is_mineru_output` decides by location: only `paper.md` and `.md` files directly under `si/` (where `ocr.ocr_artifact` has MinerU write its markdown beside each PDF) get the default source; roles are unchanged. Evidence: `test_mineru_source_is_given_by_location_not_by_role` in [[tests.torchcell.literature.test_backfill]]. Existing on-disk manifests change only when rebuilt with `--force`.

## 2026.09.30 - MinerU sidecars under si/ are OCR roles

Issue #564. `_role_for` tested the loose-`si/` rule before the MinerU rules, so the sidecars `_run_mineru.py` copies next to each SI PDF (`si/si1_content_list.json`, `si/si1_middle.json`, `si/images/*.jpg`) were roled `si_data`. The `images/` and `_content_list.json` / `_middle.json` rules now run after `si/si_data/` and before the `si/` rules, so those files are `ocr_layout` / `ocr_image` like their paper counterparts; `si/si_data/` and loose SI tables keep `si_data`. Evidence: `test_role_for_mineru_sidecars_under_si_are_ocr_roles` and `test_captured_key_full_role_table` in [[tests.torchcell.literature.test_backfill]]. Read-only count on the live mirror (605 manifests): a `--force` backfill would re-role 882 recorded entries in 38 keys (778 `si_data` to `ocr_image`, 104 `si_data` to `ocr_layout`); existing manifests are untouched until then.
