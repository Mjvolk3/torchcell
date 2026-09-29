---
id: nuyknikcnobfmyd23p3vj3p
title: Test_runners
desc: ''
updated: 1790648001514
created: 1790648001514
---

## 2026.09.28 - The dataset verification runners on tiny LMDB stores (Phase 9)

27 tests. Each `run_*` runner reads a store built under `tmp_path` in the loader's shape (pickled `model_dump()` dicts under keys `"0".."N"`, an optional sibling `interned` environment with `$ref` pointers), with the dataset registry narrowed by monkeypatch, `_genome` and `_sgd_gene_set` stubbed, and the written `preprocess/verification_report.json` read back and compared exactly: result names in order, details dicts, messages, return booleans, report paths and stdout. `load_records` returns LMDB byte order (`0, 1, 10, 2, ..., 9` for eleven keys), and the L0 failure indices refer to that order. `run_all` calls every family in `RUN_ALL_ORDER` without short-circuiting; `main()` returns the shell code, prints the banner and requires `DATA_ROOT`. The registry test freezes all 33 dataset roots, every `expected_count`, the five streaming datasets, both `background_genes` sets, the `reference_centered` and `allow_duplicate_orfs` flags, the two 0.90 floors and `SGD_GENE_FASTAS`, so a silent edit to an oracle fails CI. `stream_records` is shown to close its environment on early exit by capturing the environment `lmdb.open` returned and asserting `begin()` raises after `generator.close()` (a second open of the same path with `lock=False` succeeds even while the first is open, so that is the only observable).

Findings pinned: `run_visual_score`, `run_metabolite` and `run_protein` pass `other_name="scmd_ohya2005"` with the dataset's own gene set into `_l4_gene_containment`, so the L4 row is named `gene_containment_scmd_ohya2005` and its message labels the direction backwards (`run_morphology` names the other dataset correctly); `run_visual_score` uses `expected_count=len(records)`, so its L1 count row can never fail (line 432); the streaming environment-response branch indexes `spec["expected_count"]` (KeyError before any store is opened) while the eager branch defaults it (lines 1443 and 1455); the `model_copy(update={"name": "gene_containment_sgd"})` at lines 359 to 361 is a no-op rename. Coverage: `runners.py` 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
