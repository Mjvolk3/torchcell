---
id: ug54qe4xwaf1tud6hbaj1oj
title: Test_caudal2024_synthetic
desc: ''
updated: 1790562875426
created: 1790562875426
---

## 2026.09.27 - The Caudal 2024 loader through a real genomes registry

Ten tests. `DATA_ROOT` points into `tmp_path` with a real `GenomeManifest` and minimal FASTAs, so `registry.resolve` and its sha256 check stay on the build path; `download()`, `process()`, the digest-mismatch and missing-file paths and the parquet-cache rebuild are covered. Loader coverage 94%. Finding: `pubmed_id="38778243"` while `pubmed_url` points to PubMed 38862621 (caudal2024.py lines 684 to 685). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
