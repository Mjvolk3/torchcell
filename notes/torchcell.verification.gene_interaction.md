---
id: e2v7kem9m7vc4gthblfqm1k
title: Gene_interaction
desc: ''
updated: 1791686808179
created: 1791686808179
---

## 2026.10.10 - Gene-interaction family verifier (#889)

Single-pass L0-L4 battery for the five yeast SGA interaction stores (Costanzo 2016 DMI, Kuzmin 2018/2020 DMI and TMI). Rows in order: `structural` and `reference_structural` (each record validated as the declared `GeneInteractionExperiment` / `...Reference`), `count`, `pair_uniqueness` (the fitness family's strain and environment signatures, so `screen_id` keeps Kuzmin 2020's main and pilot screens apart), `interaction_order` (2 or 3 distinct genes at graph level `edge` / `hyperedge`), `score_finite`, `p_value_in_unit_interval`, `released_values` (stored (score, p-value) multiset against the sha256-pinned release, sorted and compared position by position, NaN equal to NaN), `reference_zero`, `reference_environment`, `signed_unclamped`, the caller's provenance audits, the shared rules, and `fitness_companion_containment` (every key is a record of the companion DMF/TMF store, by `pair_key`).

Memory: `pair_counts` holds 16-byte digests (`common.key_digest`), so the 20,705,612-key Costanzo store stays a few GB.
