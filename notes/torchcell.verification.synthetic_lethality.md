---
id: o474e61ekg7tqg21v5ci3iw
title: Synthetic_lethality
desc: ''
updated: 1791686823821
created: 1791686823821
---

## 2026.10.10 - SynLethDB pair verifier (#889)

One battery for both SynLethDB stores; `PairLabelKind` carries the class and field names (`SYNTHETIC_LETHALITY`, `SYNTHETIC_RESCUE`). Rows: `structural`, `reference_structural`, `count`, `two_distinct_genes`, `unordered_pair_uniqueness`, `positive_label`, `score_in_unit_interval` when the source defines the score (SL: the normalized confidence score) or `score_defined_by_source` when it does not (SR: no stated score is allowed), `released_rows` (the pinned CSV's kept rows, sides resolved by Entrez id), `release_accounting` (released = stored + ledgered drops, and = the paper's stated count where it states one), `reference_negative`, caller rows (`drop_ledger`, provenance audits), shared rules.
