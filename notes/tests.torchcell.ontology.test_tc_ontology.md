---
id: b135zg6jtlzad6pgklyy6zw
title: Test_tc_ontology
desc: ''
updated: 1790415131509
created: 1790415131509
---

## 2026.09.26 - print_schema_mappings on a two-node, one-edge schema

The module header called it untested; it is the tooling behind `make tc-onto`. A schema with one explicitly mapped node, one auto-mapped node (`gene`), one unmapped node, one mapped and one unmapped edge, and a stray non-dict entry is written to `tmp_path`; the compact and expanded outputs are checked line by line, including the summary arithmetic `1/3 explicit + 1 auto-mapped = 2/3 total` and the sorted Biolink concept list. Row labels are padded to 25 characters and the warning sign is two code points, so the expected rows are built with the same format expression. biocypher is imported inside the tests after `chdir(tmp_path)` because its import writes a log directory into the working directory. `scripts/test_quality_check.py` now treats pytest's capture fixtures as observed outputs, which is what made these stdout assertions pass the anti-padding lint. Phase 3 of [[plan.test-suite-buildout.2026.09.25]].
