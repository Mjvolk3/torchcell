---
id: 7sbzyk85y85igbj7r1d2joa
title: Test_label_policy
desc: ''
updated: 1790765108562
created: 1790765108562
---

## 2026.09.30 - Phase 14: precedence, Stouffer and inverse variance in closed form

Twenty to thirty-seven tests, 83 to 100 percent. `source_key` for kuzmin2020 and the converted-zero check running first with both refusal messages; `standard_error` = sd/sqrt(n) and its fallbacks; `select` on an absent label and the unknown-label message; which source wins when a higher-ranked one is absent (the full `LabelChoice`) and a refused converted zero still counting in `n_entries_available`; the mean fallback when one replicate lacks an sd (se 0.005, sd 0.01), `first`, `min`, `none`; Stouffer 0.0010020 at equal weights, 0.0101986 at weights 1 and 9, 0.0398580 when the screens disagree, unusable p-values skipped; year promotion; `entries_from_records` on both key spellings. Existing tests tightened to exact values (inverse variance 10080/10100 and 1/sqrt(10100); trigenic tau 0.45441 and 0.455171).

Findings: `int()` truncates the temperature, so 29.9 keys as `costanzo2016@29` (line 101); `select_double` averages strain-matched entries across different sources (374-384); any experiment type without "interaction" in its name is read as fitness (446); a `temp=None` key hides a filled `temperature` (447).
