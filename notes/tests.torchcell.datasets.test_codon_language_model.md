---
id: xccad405tgu1wczussvm7fo
title: Test_codon_language_model
desc: ''
updated: 1790780573870
created: 1790780573870
---

## 2026.09.30 - Phase 18: CaLM through a stand-in module

New file, four tests, 19 to 90 percent alone; CaLM is faked through a stand-in `sys.modules["calm"]` since the dataset imports it lazily. The CDS-versus-truncated-window branches with exact sequences, stored values, caching. Findings: the `is_max_size` flag is never passed on, so YAL003W's window is [2000, 5072) rather than the symmetric [2014, 5086); for a spliced gene the stored window has the CDS length but sits on the locus, so it is not the sequence that was embedded.

## 2026.09.30 - Findings retired (issue #543)

Retired: the locus window stored for a spliced gene and the unused `is_max_size` flag. Now asserted: `dna_windows["calm"]` is exactly the embedded string (YAL001W `chrI[100:112]`, YAL002C `ATGGCCTAA`, YAL003W the first 3,072 nt of its CDS); chunked builds consume `calm.partial.pt`; a stale chunk file is removed with the exact warning and the store holds the three genes only; the spaced message.
