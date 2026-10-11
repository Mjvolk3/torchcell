---
id: o474e61ekg7tqg21v5ci3iw
title: Synthetic_lethality
desc: ''
updated: 1791686823821
created: 1791686823821
---

## 2026.10.10 - SynLethDB pair verifier (#889)

One battery for both SynLethDB stores; `PairLabelKind` carries the class and field names (`SYNTHETIC_LETHALITY`, `SYNTHETIC_RESCUE`). Rows: `structural`, `reference_structural`, `count`, `two_distinct_genes`, `unordered_pair_uniqueness`, `positive_label`, `score_in_unit_interval` when the source defines the score (SL: the normalized confidence score) or `score_defined_by_source` when it does not (SR: no stated score is allowed), `released_rows` (the pinned CSV's kept rows, sides resolved by Entrez id), `release_accounting` (released = stored + ledgered drops, and = the paper's stated count where it states one), `reference_negative`, caller rows (`drop_ledger`, provenance audits), shared rules.

## 2026.10.11 - Module removed: SynLethDB is a provenance record

`Environment.media` stays required (owner, [[torchcell.datamodels.schema]]) and no SynLethDB field states a growth condition, so both SynLethDB classes left the served graph and their source is a provenance record ([[torchcell.datasets.scerevisiae.synth_leth_db]]). There are no synthetic-pair stores left to verify: `torchcell/verification/synthetic_lethality.py` and `runners.run_synthetic_pairs` are removed, with `SYNTHETIC_PAIR_DATASETS`, `released_synlethdb_pairs` and their tests. `run_all` no longer calls the family.

What the verifier checked moved to the provenance record: the release inventory (0 condition columns, `r.cell_line` empty on all 20,948 rows), the per-source decomposition (1,741 SL and 1,918 SR source PubMed ids) and the score measurement (every SL score is a function of the evidence label; the 74 SR scores have no sourced definition) are measured functions pinned against the two released files' sha256.
