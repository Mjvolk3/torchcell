---
id: ihkm6t3bqhnjmc2nq0mmaqn
title: Enzyme_kinetics
desc: ''
updated: 1790815005097
created: 1790815005097
---

## 2026.09.30 - Order-independent median tie-break

Issue #525. Previously the last rung took `statistics.median` and returned the first row nearest to it, so with an even number of tied rows ([1.0, 3.0], median 2.0) the input order decided (1.0 forward, 3.0 reversed), against the module docstring. Now the rung takes `statistics.median_low`, which is always one measured value (keeping the row's PubMed id, temperature and pH traceable), and among rows sharing that value picks the one whose `model_dump_json()` sorts first. Odd counts are unchanged. Evidence: `test_even_number_of_ties_takes_the_lower_median_in_any_row_order` and `test_rows_sharing_the_median_value_resolve_to_the_same_row_in_any_order` in [[tests.torchcell.metabolism.test_enzyme_kinetics]].

Review follow-up: the order among rows sharing the median value is lexicographic on the `model_dump_json()` text (so `"ph":10.0` sorts before `"ph":9.0`), deterministic by construction. The lower median is conservative for a `k_cat` capacity bound but raises saturation for a `K_M`. An unequal-multiplicity case `[1.0, 3.0, 3.0, 3.0]` resolves to 3.0 in any order.
