---
id: td91f3egmjvuh6ygvux31yq
title: Test_cellpose_seg
desc: ''
updated: 1790649903099
created: 1790649903099
---

## 2026.09.28 - The Cellpose pipeline on precomputed masks, no model (Phase 9)

30 tests. `load_cellpose_model` is exercised against a monkeypatched `CellposeModel` (no weights, no instantiation; `import cellpose.models` only computes a path string); the pipeline runs on `precomputed_masks` with `model=None`, plus one test that a fake model's `eval` is called when no masks are given. Lattice fitting (center (150, 209), angle exactly 0 since 12 of 14 pair angles are 0), relaxation with extrapolation (10, 21 to 32), the exact homography and its clamp, sheared lattices within the 15-px assignment radius, edge-row snapping (gain 8 accepted, 4 refused), instance properties (27 px, centroid (2, 5), circularity 4 pi 27 / 400), tightening (49 px kept against the 121 halo since 49 + 4 * 7 = 77 and 49 < 60.5), recovery (crop 54..146, depth 23, None at 8 and 0; the recovered disk grown by `disk(3)` is 1001 px, pinned numerically since a digital disk of radius 18 has 1009), contrast, overlays with thick boundaries (52 and 20 px), the table with the 45.69-px N assignment and the kept-color map, the off-grid and no-gel paths, `multi_min_frac`, and the default configuration refusing a small plate. The shared rotation-sign finding is in [[tests.torchcell.sga.test_image]]. Coverage 0% to 95%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
