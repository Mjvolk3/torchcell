---
id: 6hmyfrwpj4itlcetokw8i6p
title: Test_regression_to_classification
desc: ''
updated: 1790765116116
created: 1790765116116
---

## 2026.09.30 - Phase 14: the remaining strategies in closed form

Twenty-two to forty-two tests, 85 to 99 percent (line 428 is unreachable, see the last finding). Robust scaling round trip with exact stats; the all-NaN pass-through; auto bins giving edges [0, 2, 4]; normalized edges and their denormalized copy; exact one-hot, soft and ordinal outputs; the seeded categorical and ordinal inverses (the seed-42 draws are measured); the windowed soft inverse (2.625, 1.5, 0.5); every error message; the `InverseCompose` repr.

Findings: the constructor accepts an unknown normalization strategy and only `normalize` refuses it (lines 63, 88, 108); a soft-label row whose Gaussians underflow stays all zeros instead of summing to 1 (217); the soft inverse is deterministic and edge-dependent although the docstring says "random sampling" (469-500); an unknown `label_type` silently leaves the label unbinned going forward and returns NaN from the inverse (389-407, 446-523); `inverse` reads `.device` at 418 before the list-to-tensor conversion at 427-428, so a list input raises `AttributeError`.
