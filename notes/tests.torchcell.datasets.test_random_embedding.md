---
id: lxg705rp06vj7s4vk42pypd
title: Test_random_embedding
desc: ''
updated: 1790780581471
created: 1790780581471
---

## 2026.09.30 - Phase 18: seed determinism, chunking, the RNG side effect

New file, eight tests (10 cases), 21 to 92 percent alone. Rows are the seed-42 stream whatever the caller seeded, first row [0.8823, 0.9150, 0.3829, ...]; `random_1` and `random_6579` windows; chunked saves equal to one chunk; a second construction never reseeds (the proof that `process` did not run); each test restores the global torch RNG state afterwards.

Findings: construction reseeds the global torch RNG, so the caller's next `torch.rand(1)` is 0.9593; a chunk file left by an interrupted build blocks the next construction with `ValueError: not enough values to unpack (expected 2, got 1)`, so interrupted builds never resume; the base error message is missing a space (`'...'.Valid options are:`); a spliced gene's window sits on the locus, not the CDS.

## 2026.09.30 - Findings retired (issue #543)

Retired: global RNG reseeding and the blocking chunk file. Now asserted: after a build the caller's next `torch.rand(1)` is seed 123's first draw; the rows are still the seed-42 stream; chunked builds consume `random_10.partial.pt`; a stale chunk file is removed with the exact warning and the store is the three-gene seed-42 stream; the spaced message.
