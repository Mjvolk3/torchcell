---
id: lk6zcj4yolx2eskmksdvunp
title: test-suite-buildout-p19
desc: ''
updated: 1790920901132
created: 1790920901132
---

## 2026.10.02

- [x] PR-19 of [[plan.test-suite-buildout.2026.09.25]]: the NaN-tolerant metrics ([[tests.torchcell.metrics.test_nan_tolerant_metrics]], [[tests.torchcell.metrics.test_nan_tolerant_classification_metrics]]), the Dango model and trainer ([[tests.torchcell.models.test_dango]], [[tests.torchcell.trainers.test_int_dango]]), the hetero-cell trainer ([[tests.torchcell.trainers.test_int_hetero_cell]], [[tests.torchcell.trainers.test_int_hetero_cell_diffusion]]), the DCell trainer and the optimized DCell against the reference ([[tests.torchcell.trainers.test_int_dcell]], [[tests.torchcell.models.test_dcell_opt]]), the adapter, raw-query and Dango GI remainders ([[tests.torchcell.adapters.test_cell_adapter]], [[tests.torchcell.data.test_neo4j_query_raw]], [[tests.torchcell.models.test_hetero_cell_bipartite_dango_gi]]); plain pytest hides the GPUs unless `--gpu` ([[tests.conftest]], [[tests.torchcell.test_conftest_no_gpu]]) and collection is confined to `tests/`; 270 new test functions, two independent Opus 5.5 audits (246 reviewed, 0 rejected, 6 rewritten, 39 of 43 findings confirmed and 4 corrected), TOTAL 58.4% to 61.7%; record in [[test-campaign.2026.09.25]]
