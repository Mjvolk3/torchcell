---
id: q2m1mo8ytk1bggiq6asjou9
title: Compound_curator_726_860
desc: ''
updated: 1791698124017
created: 1791698124017
---

## 2026.10.11 - The #726 / #860 curator pass

Runs `torchcell.datamodels.compound_identity_curate.curate` on
`compound_identity_inputs/bacterial_conditions.txt` alone and merges the rows into the
pinned table additively: a label whose CID an existing row holds is folded into that row as
a synonym (InChIKeys must agree), a new row must be named by an input label, and no lookup
key may be claimed twice. It also re-reads Price 2018 Tables S2-S4 (`si3.xlsx`, sha256
re-checked) and looks up every CAS number they state for a curated label.

Outputs: `results/compound_curator_726_860.csv` (one row per label: status, row name,
InChIKey, CID, ChEBI, directive, fold target, PubChem Title and SMILES of the adopted
record, endpoint, Price CAS and whether it agrees, stores, reason),
`results/compound_curator_726_860.json` (the same plus the summary), and
`results/compound_curator_726_860_log.txt`.

Measured outcome (2026-10-10 run, table `5a476def...` -> `d45d3ded...`): 178 labels, 166
RESOLVED, 4 RESOLVED_MIXTURE, 3 UNDEFINED_MIXTURE, 5 UNRESOLVED_PUBLIC; 146 rows added, 26
labels folded; 63 CAS checked, 54 agree. Interpretation and the per-label decisions:
[[torchcell.datamodels.compound_identity]] (2026.10.11).
