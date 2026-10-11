---
id: xhn0j8haqymbkiibo472s2b
title: Yeast_zotero_collection_audit
desc: ''
updated: 1791728050095
created: 1791728050095
---

## 2026.10.11 - The yeast Zotero collection audit as data

`experiments/database/scripts/yeast_zotero_collection_audit.py` holds the audit of the 30 items of the Zotero collection `database/Saccharomyces-cerevisiae` (`KCG3NWH7`) that had no loader on 2026-10-11, one typed `AuditEntry` per key (category a to g, released artifacts, measured record count, genotype basis, phenotype class or gap, overlap, verbatim quotes). It gates the 20 dataset candidates with the `torchcell.candidates.gates` evaluators and, with `--write`, stores the verdicts in `database/candidates/` and the note tables in `experiments/database/results/yeast_zotero_collection_audit.md` (gitignored; copied into [[database.yeast-zotero-collection-audit.2026.10.11]]).

Where the deterministic gates cannot read a judgment off a row, the gate is composed from the same record types and its reason starts `composed`: a host the paper names but the row does not (`HostSpec`, probed through the `gates.HOSTS` token, or `absent_provisionable` when no tier set exists), a release whose per-record values are walled or were never released (G3 `blocked:wall` with the recipe), and a gap beside an existing phenotype class (G5 `gap` with its issue). Off-schedule keys get a `RowSpec`. Every quote is re-read by `python -m torchcell.candidates audit` (79 ok on 2026-10-11).

Run: `PYTHONPATH=$PWD python experiments/database/scripts/yeast_zotero_collection_audit.py --write`.
