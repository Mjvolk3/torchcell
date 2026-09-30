---
id: 9tkh1vs1ftmg0sxg3i78er4
title: Test_ozaydin2013
desc: ''
updated: 1790765138766
created: 1790765138766
---

## 2026.09.30 - Phase 14: the file no longer skips at import

One to twelve tests, 0 to 95 percent for this file alone (with the sibling `_synthetic` file the module was at 84). The import-time `load_dotenv()` and the file-level skip are gone; the mirror marks sit on the smoke test only. `download()` at the 9,999-byte and 10,000-byte boundaries with exact messages, the ledger log lines, an out-of-scale score of 6 refused, `main()`.

Findings: when one ORF is scored on two strains the first row's strain wins (lines 251-263); two equal scores give `visual_score_min` 2.0, not None; the medium is a free-text `Media(name="SC-URA")` stub with no components (361), not the sourced `media.SC_URA`; the three cassette genes have no `plasmid_contig_id`, `locus_tag` or `integration_locus` (the sibling file compared the cassette against `_carotenogenic_cassette()` itself, so its fields were never pinned independently). No GitHub issue tracks the Ozaydin findings yet.
