---
id: lru4ahw1a73k5ovdxtykvih
title: Synthetic_token_offset
desc: ''
updated: 1790417095940
created: 1790417095940
---

## 2026.09.26 - A second source that differs by a known offset

Clones every entry row of the configured labels with the value plus `delta` (normalized units when run after the normalizer) under a synthetic token slot, or under the original token in the control. Emits `phenotype_synthetic` so the smoke check can tell clones from originals when the token no longer can. Used only by `cgt_030_smoke_*`; the inverse compose excludes it. Tests: `tests/torchcell/transforms/test_synthetic_token_offset.py`.
