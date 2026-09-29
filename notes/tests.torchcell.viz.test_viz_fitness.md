---
id: m4b6bn4oyju5ga4xln4gyza
title: Test_viz_fitness
desc: ''
updated: 1790650184187
created: 1790650184187
---

## 2026.09.28 - The fitness box plot on simulated data (Phase 9)

7 tests. Binning of predictions, Pearson 0.997, Spearman 1.000 and R squared 0.993 recomputed on float32, the median markers at `i + 0.036` and `i + 0.964`, the two black wild-type reference lines, the N/A titles for one valid pair or a constant measured vector (scipy 1.16 returns NaN with a `ConstantInputWarning` for all three statistics, so the `except ValueError` arms at lines 45 to 51 never run; what is pinned is the NaN to "N/A" formatting), the simulated data's shapes, clip bounds, `n // 20` special cases and seed-0 NaN counts, and `main()` building one figure. The header comment at line 4 points the test file at `torchcell/viz/test_fitness.py` instead of `tests/torchcell/viz/`; the test file is `test_viz_fitness.py` because a second `test_fitness.py` basename (the verification one) breaks collection of the full suite, and the pair is registered in `pyproject.toml`. Coverage to 97%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]].
