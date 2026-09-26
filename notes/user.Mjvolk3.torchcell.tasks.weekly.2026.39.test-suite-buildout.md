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
- [ ] PR-0b: `tests/torchcell/test_import_all.py`, `scripts/coverage_gaps.py`, campaign note with the Phase 0 table
