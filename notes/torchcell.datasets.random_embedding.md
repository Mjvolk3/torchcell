---
id: tz5zjmbgqvk0egpqueq0p3r
title: Random_embedding
desc: ''
updated: 1716335486876
created: 1716335486876
---

## 2026.09.30 - Private generator and chunk file (issue #543)

- `process` called `torch.manual_seed(42)`, reseeding the caller's global RNG. It now draws from `torch.Generator().manual_seed(42)`; on CPU this is the same seed-42 stream, so the stored rows are unchanged and the caller's stream continues.
- Chunks go to `processed/<name>.partial.pt`; a stale chunk file is removed with a logged warning. A chunk file at the final store path from an interrupted build made by the OLD code still blocks construction; move it aside by hand.
- Left open: `dna_windows` still holds `window(len(cds))` on the locus for a short gene, which is not the CDS for a spliced gene; nothing is embedded from it, so it is metadata only.

A rebuilt store is identical on disk (same rows, same windows). Tests: [[tests.torchcell.datasets.test_random_embedding]].
