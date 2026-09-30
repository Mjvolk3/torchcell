---
id: vvkceo35ma4rd2xmh19w9ts
title: Test_hetero_to_dense_mask
desc: ''
updated: 1790773255759
created: 1790773255759
---

## 2026.09.30 - Phase 16: out-of-range indices, missing edge types, per-node padding

Five to nine tests, 93 to 100 percent. Out-of-range indices, an edge type with no index, `pos` and per-node tensor padding. Findings: a square per-node matrix is padded on rows only (line 141); an edge into the padding row passes the validity filter (63-69).
