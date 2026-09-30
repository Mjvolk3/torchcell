---
id: b135zg6jtlzad6pgklyy6zw
title: Test_tc_ontology
desc: ''
updated: 1790415131509
created: 1790415131509
---

## 2026.09.26 - print_schema_mappings on a two-node, one-edge schema

The module header called it untested; it is the tooling behind `make tc-onto`. A schema with one explicitly mapped node, one auto-mapped node (`gene`), one unmapped node, one mapped and one unmapped edge, and a stray non-dict entry is written to `tmp_path`; the compact and expanded outputs are checked line by line, including the summary arithmetic `1/3 explicit + 1 auto-mapped = 2/3 total` and the sorted Biolink concept list. Row labels are padded to 25 characters and the warning sign is two code points, so the expected rows are built with the same format expression. biocypher is imported inside the tests after `chdir(tmp_path)` because its import writes a log directory into the working directory. `scripts/test_quality_check.py` now treats pytest's capture fixtures as observed outputs, which is what made these stdout assertions pass the anti-padding lint. Phase 3 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.09.30 - Phase 15: the compact table of the real schema

Three to ten tests, 86 to 100 percent. The real schema's full compact table (26 nodes, 13 edges, 10 concepts; this test changes with every schema edit, deliberately); a fully mapped schema in both formats prints no warning and skips node properties; a schema with neither auto-mapped nor unmapped nodes prints neither line; the BioCypher printers delegate with `offline=True` through a recorder; `main` argument handling.

Findings: the compact headers are hardcoded "NODES (16 total)" and "EDGES (11 total)" while the real schema has 26 and 13 (lines 147, 160); list-valued edge endpoints print as Python list reprs (234-236).
