---
id: 2iimjjftnskamyyo1sytz27
title: Test_normalize
desc: ''
updated: 1790649918452
created: 1790649918452
---

## 2026.09.28 - Spatial and plate normalization on hand-built grids (Phase 9)

9 tests: multiplicative correction (plate median 40, factors 0.625, 1.25, 2.5), row and column factors with a dropped row (median 35, factors 25/35 and 15/35), a step plate where every window median is on its own side, the plate-median fallback with few neighbors (20/3, 60), flags, the cap and the jackknife (MAD 0.1, z 6.745, flag order "JK;CP"), the optional C flag, no strain, the zero-MAD skip (z 13.49) and status order; the normalization 1600 / 50 = 32 to 1.0. Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
