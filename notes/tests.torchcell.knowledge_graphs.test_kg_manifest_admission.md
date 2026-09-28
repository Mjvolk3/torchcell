---
id: d8pr0oja7i6sk0uejx5comv
title: Test_kg_manifest_admission
desc: ''
updated: 1790562642944
created: 1790562642944
---

## 2026.09.27 - Admission verdicts on a miniature checkout

Twenty-eight test functions on a miniature checkout under `tmp_path` with one served and one new toy dataset: the baseline manifest comes from `bootstrap_manifest`, and each test changes one thing. Git is a PATH-stubbed script, Neo4j a scripted fake at `neo4j.GraphDatabase.driver`, and the clock (`kg_manifest._now`) is pinned. Every drift kind is tested for both verdicts with the exact reason text, plus the superset lineage, the event log and all four CLI subcommands through `main(argv)`. Module coverage from this file 95%, 99% with the existing tests. Finding: the bootstrap records `cell_adapter.py` among the adapter files but the list the manifest adopts after an admission excludes it, so that key disappears at the first admission (kg_manifest.py line 714 against 522). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
