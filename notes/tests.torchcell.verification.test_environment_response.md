---
id: c7s67075t0zi36t0bs4wqri
title: Test_environment_response
desc: ''
updated: 1790765146297
created: 1790765146297
---

## 2026.09.30 - Phase 14: exact result rows for both verifiers on a hand-built release

New file, fifteen tests, this file alone 99 percent of `torchcell/verification/environment_response.py` (the sibling directory tests held it at 81). Exact result rows for a passing release and for each failure mode on both the eager and the streaming verifiers, the `summary()` lines, the strain and condition signatures, the gene-set helper. The module has no `main`.

Findings: the two verifiers disagree although the streaming docstring calls them "semantically identical": three copies of one record count as 1 duplicate (eager) against 2 (streaming), with different wording; bad-value indices differ (eager indexes among non-missing values and adds a `reason`, streaming indexes by record); the measurement-type message prints the enum repr.

## 2026.10.01 - Findings retired: eager and streaming agree (issue #529)

Retired the three Findings (duplicate count and wording, bad-value indexing, enum repr). Now asserted for BOTH entry points: `pair_uniqueness` counts redundant records (three copies are 2, message `2 records duplicate an earlier (study, strain, condition) triple; 3 unique triples`); bad values carry the RECORD index and a `reason`; `measurement_type_consistent` prints `'log2_ratio'` and lists `['log2_ratio', 'z_score']`. New `test_eager_and_streaming_reports_are_equal` runs both verifiers over one hand-built release that fails every own rule and asserts the reports equal result by result, field by field, and as whole models. 15 to 16 tests.
