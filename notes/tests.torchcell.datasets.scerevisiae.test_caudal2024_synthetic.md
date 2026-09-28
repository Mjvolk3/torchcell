---
id: ug54qe4xwaf1tud6hbaj1oj
title: Test_caudal2024_synthetic
desc: ''
updated: 1790562875426
created: 1790562875426
---

## 2026.09.27 - The Caudal 2024 loader through a real genomes registry

Ten tests. `DATA_ROOT` points into `tmp_path` with a real `GenomeManifest` and minimal FASTAs, so `registry.resolve` and its sha256 check stay on the build path; `download()`, `process()`, the digest-mismatch and missing-file paths and the parquet-cache rebuild are covered. Loader coverage 94%. Finding: `pubmed_id="38778243"` while `pubmed_url` points to PubMed 38862621 (caudal2024.py lines 684 to 685). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
