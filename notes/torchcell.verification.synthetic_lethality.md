---
id: o474e61ekg7tqg21v5ci3iw
title: Synthetic_lethality
desc: ''
updated: 1791686823821
created: 1791686823821
---

## 2026.10.10 - SynLethDB pair verifier (#889)

One battery for both SynLethDB stores; `PairLabelKind` carries the class and field names (`SYNTHETIC_LETHALITY`, `SYNTHETIC_RESCUE`). Rows: `structural`, `reference_structural`, `count`, `two_distinct_genes`, `unordered_pair_uniqueness`, `positive_label`, `score_in_unit_interval` when the source defines the score (SL: the normalized confidence score) or `score_defined_by_source` when it does not (SR: no stated score is allowed), `released_rows` (the pinned CSV's kept rows, sides resolved by Entrez id), `release_accounting` (released = stored + ledgered drops, and = the paper's stated count where it states one), `reference_negative`, caller rows (`drop_ledger`, provenance audits), shared rules.

## 2026.10.11 - No released score where the source defines none (#903)

The SR loader stores each of the 74 released `r.statistic_score` values as a typed gap, since no mirrored source defines them ([[torchcell.datasets.scerevisiae.synth_leth_db]]). `runners.released_synlethdb_pairs(..., keep_scores=False)` reads every released score as None for a file whose registry entry has no `score_definition`, so `released_rows` compares what `score_defined_by_source` already requires; on the as-written runner the 74 gapped records were its only mismatch. Both SynLethDB stores now pass every rule.
