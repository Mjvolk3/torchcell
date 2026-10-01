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

## 2026.10.01 - The four findings retired (issue #527)

The four pinned findings are fixed in [[torchcell.data.label_policy]] and the tests now assert the corrected contract. `test_source_key_refuses_a_fractional_costanzo_temperature`: 30.0 and 26 key exactly, 29.9 and NaN raise with the exact message. `test_strain_matched_doubles_are_ranked_by_source_not_averaged_across_it`: kuzmin2018 0.50 wins by default (1 combined of 2 matches), kuzmin2020 0.70 once promoted, and two kuzmin2018 matches still average to 0.55. `test_a_strain_match_from_no_listed_source_falls_to_the_pairs_own_entries`: an unlisted costanzo2016@26 match yields to kuzmin2020 0.81. `test_only_fitness_and_gene_interaction_types_fill_a_label`: the two schema types map, "calmorph", "gene_interaction" and "gene essentiality" are refused. `test_entries_from_records_reads_the_long_key_when_the_short_one_is_none`: `temp=None` reads `temperature=30`, `p=None` reads `p_value=0.01`, equal values under both spellings read once, conflicting ones are refused. The both-spellings test now uses the stored type "gene interaction".
