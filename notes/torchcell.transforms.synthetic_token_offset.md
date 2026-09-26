---
id: lru4ahw1a73k5ovdxtykvih
title: Synthetic_token_offset
desc: ''
updated: 1790417095940
created: 1790417095940
---

## 2026.09.26 - A second source that differs by a known offset

Clones every entry row of the configured labels with the value plus `delta` (normalized units when run after the normalizer) under a synthetic token slot, or under the original token in the control. Emits `phenotype_synthetic` so the smoke check can tell clones from originals when the token no longer can. Used only by `cgt_030_smoke_*`; the inverse compose excludes it. Tests: `tests/torchcell/transforms/test_synthetic_token_offset.py`.

## 2026.09.26 - Twin per source, not one token for everything

Rewritten after GilaHyper jobs 2863 and 2866: `token_map` maps each source token to its own synthetic twin, so a twin's rows are exactly the source's rows shifted and its optimal bias is the source's plus `delta` at any training stage. Job 2867 measured 0.285 for a 0.3 offset with the twin design where the single-token design had measured 0.79 to 0.93.
