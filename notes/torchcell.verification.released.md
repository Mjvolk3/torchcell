---
id: cpjkucn9ao32asi6x6uzlfo
title: Released
desc: ''
updated: 1791686831607
created: 1791686831607
---

## 2026.10.10 - Sha256-pinned released-table readers (#889)

The family verifiers compare a store to the table its paper released, read here rather than through the loader. `ReleasedFile` pins a file name to the sha256 the served build consumed; `drifted_files` reports drift (the caller fails the rule on it, nothing is read). `interaction_values_from_table` reads the score and p-value columns by their released header names (`COSTANZO2016_SCORE`, `KUZMIN_FINAL_SCORE`, ...), tab-delimited or xlsx with the title row skipped, filtered by `Combined mutant type` for Kuzmin. `sorted_value_mismatches` compares two multisets order-free. `sgd_json_digest` and `sgd_inviable_null_annotations` cover the SGD per-gene JSON release.
