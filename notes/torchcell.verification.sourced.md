---
id: q5fjktr0a1hqozvx894br3q
title: Sourced
desc: ''
updated: 1783563843099
created: 1783563843099
---

## 2026.07.08 - SourcedValue: bind every hardcoded constant to a sha256-pinned verbatim quote

This module is the mechanical enforcement of torchcell's provenance-first principle: any hardcoded number a loader uses (an `n_samples`, a threshold) must trace to the exact paper artifact it came from -- `citation_key` + path inside the MinerU `torchcell-library` + file `sha256` + a verbatim `quote`. `SourcedValue` makes that binding a validated pydantic object rather than a code comment, so a constant can never drift away from its justification. Reading `.value` needs nothing but Python (the ML env never touches MinerU); the `audit_sourced_value` re-check is opt-in and skipped when the library isn't mounted, so CI stays green.

- The file `sha256` -- not a line number -- is the reproducibility anchor: OCR output reflows when the extractor updates, so a line reference silently rots while the hash detects any re-OCR and the quote substring re-locates the value.
- The model validator REQUIRES `citation_key`, `sha256`, and a non-empty `quote` at construction: an unauditable sourced value is impossible to build.
- Complements [[torchcell.verification.report]]'s `StatDerivation`, which records the DERIVED (back-solved / range-fallback) statistics that no quote can back -- sourced vs derived are the two provenance stances.
- The audit returns an L3 `LevelResult` (see [[torchcell.verification.levels]]), so provenance verification composes into the same report as the record-level gate.
