---
id: g21rhvbj8dfakhnrh3y033q
title: test-suite-buildout
desc: ''
updated: 1790376915792
created: 1790376915792
---

## 2026.09.25

- [x] Plan the test-suite build-out: root conftest + flag-gated markers, parametrized import-all, contract/invariant sweeps, live-model and trainer CPU tests, hermetic script tests, behavioral-coverage reporting with a no-cheating quality audit, diff-cover + paired-test gates on new code, legacy move to torchcell/legacy/ for hard mode, coverage gate after Phase 1 [[plan.test-suite-buildout.2026.09.25]]

## 2026.09.26

- [x] PR-0a of [[plan.test-suite-buildout.2026.09.25]]: [[tests.conftest]] (sentinel `DATA_ROOT`, six opt-in flags, boundary guards), [[tests.torchcell.conftest]] (CGT, FakeTxn, DCell fixtures), the three gate scripts [[scripts.test_quality_check]], [[scripts.check_paired_tests]], [[scripts.legacy_partition]], `[tool.torchcell.test_exceptions]`, Makefile test/cov targets, diff-cover + paired-test + quality steps in test.yaml, mypy diff-scope on tests/, [[torchcell.metabolism]] package marker
- [ ] Confirm the computed legacy move list before PR-0d (`make legacy-table`); `models/dcell.py` and `trainers/neo_regression.py` are init-only until their Phase 2 tests land
- [x] PR-0b: [[tests.torchcell.test_import_all]] (384 modules import, 18 strict xfails, 2 more never-import modules found by the network guard), [[scripts.coverage_gaps]], the behavioral / import-only split in test.yaml with the coverage JSONs uploaded as a workflow artifact, and the Phase 0 baseline table in [[test-campaign.2026.09.25]]
- [x] PR-1 (Phase 1, contracts and invariants): [[tests.torchcell.datamodels.test_datamodels_roundtrip]] (109 models), [[tests.torchcell.datasets.test_dataset_registry]] (52 loaders; seven `raw_file_names` are a bare `str`, pinned in `STR_RAW`, fix blocked by diff-cover until the data-gated read path is covered), exact-value tests for `logcosh`, `list_mle` (never reads the true ranking as implemented; real bug in a live loss, strict xfail against the Plackett-Luce value), `losses/dcell`, `mle_wasserstein` schedulers, `point_dist_graph_reg`, `isomorphic_cell_loss`, and `HeteroToDenseMask` (its extra-attribute padding loop iterated `dir(store)` and padded nothing; fixed)
