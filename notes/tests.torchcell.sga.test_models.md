---
id: ede3gzkixo2wxbh9jnihjrg
title: Test_models
desc: ''
updated: 1790649957447
created: 1790649957447
---

## 2026.09.28 - The sga pydantic models (Phase 9)

4 tests (7 cases): `NormalizationConfig` defaults (min_size 1.0, both corrections on, radius 2 with 4 minimum neighbors) and bounds (radius and minimum neighbors at least 1, jackknife z above 0, each violation a single error at `loc == (field,)`); `StrainScore` requires strain, n_total and n_used and defaults every statistic to None; `ScoreReport` keeps its strains in order with the optional medians None. Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
