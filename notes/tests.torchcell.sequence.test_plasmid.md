---
id: 08ivxsex6lut3rnqha2q9ca
title: Test_plasmid
desc: ''
updated: 1791270289295
created: 1791270289295
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): `parse_genbank_component` on a hand-written 20 bp circular GenBank map: identity, topology, sequence, SO roles per feature type (unmapped type -> `SO:0000110`), names from `label` then `gene` then empty, 1-based closed to half-open coordinates, reverse strand, and provenance (basename, sha256 from hashlib, citation key); `_sha256` over more than one 1 MiB block.

Finding: a compound location is reduced to `[min start, max end)`, so the origin-spanning `join(18..20,1..2)` becomes `[0, 20)`, the whole plasmid (plasmid.py:200-208).

### Audit 2 notes applied

- Reach (audit 2): nothing calls `parse_genbank_component` yet, so latent.
