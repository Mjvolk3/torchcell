---
id: werjofc1zgvnoxlmqui082r
title: Test_compound_identity
desc: ''
updated: 1790777378536
created: 1790777378536
---

## 2026.09.30 - Phase 17: normalization, precedence, table refusals

Twenty-four to thirty-two tests, 95 to 100 percent. Normalization rules; a name match outranking the CID; `known_proprietary` applying only when no row matches; the unparseable-SMILES miss on both routes with exact dumps; caller fields winning over the row; `_load_table`'s sha256 refusal and key-collision refusal (`NaCl` versus `sodium chloride`) on tiny tables under `tmp_path`; a row whose own spellings fold to one key not a collision; a CID shared by two rows keeping the first. The module has no CURIE resolution.
