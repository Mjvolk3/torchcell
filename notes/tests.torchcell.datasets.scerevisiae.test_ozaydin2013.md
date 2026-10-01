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

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Retired the first-strain-wins finding (issue #528)

Retired: the first row's strain labeling an ORF scored on two strains. Now asserted: numeric scores on two strains raise `MixedStrainScoresError` with an exact message, the strain is the one the score was taken on (text row first), and one-strain replicates still aggregate max, min, n and OR-ed flags.
