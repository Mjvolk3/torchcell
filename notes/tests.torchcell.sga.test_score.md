---
id: 9t05h5fpdtzrgr1algx1kzq
title: Test_score
desc: ''
updated: 1790649926645
created: 1790649926645
---

## 2026.09.28 - Strain scores with an exact Mann-Whitney p (Phase 9)

6 tests: the SD of sqrt(1/30), the exact Mann-Whitney U p of 2 / C(7, 3) = 2/35 (n = 3, m = 4, no ties), gene B at 0.1 sqrt(2), the sort order "BY4741" before "Blank_media", the jackknife (0.45, 0.9), a zero wild type disabling ratios (`0.0 in (None, 0)`), layout errors, `_used` with and without the jackknife column, and the table columns equal to `StrainScore.model_fields`. Noted: `~df.get("is_jackknife", False)` (score.py line 34, assay.py line 29) evaluates `~False` when the column is absent, a DeprecationWarning slated for removal in Python 3.16; the current behavior is pinned. Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
