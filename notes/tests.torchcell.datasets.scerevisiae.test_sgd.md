---
id: ikofs8r4e2kj8hj39yqh0sv
title: Test_sgd
desc: ''
updated: 1790550017602
created: 1790550017602
---

## 2026.09.27 - The SGD gene-essentiality loader on a stub graph and a fake Entrez

The loader has no raw files; `process()` walks `scerevisiae_graph.G_raw` and fetches each phenotype's publication through `Bio.Entrez`. Here `G_raw` is a three-node networkx graph, `Entrez.efetch` and `Entrez.read` are replaced on `Bio.Entrez` itself (the module object the loader holds) so `get_publication_info` runs for real against canned records, `time.sleep`, `random.uniform` and `main_get_all_genes` are patched, and `DATA_ROOT` points into `tmp_path` with 100 gene JSON files. Six tests: the essentiality records with their publications by `model_dump()` equality, the side files, the retry and error paths. Loader coverage from this file 89% (an unreachable `return None` at sgd.py line 108 and `main()` remain). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
