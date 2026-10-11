---
id: o8pw5a2phimyxhbv2399eth
title: Manual_problems_ledger_measurements
desc: ''
updated: 1791693866747
created: 1791693866747
---

## 2026.10.10 - What the manual-problems ledger measures on the machine

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/manual_problems_ledger_measurements.py`. Results: `experiments/036-dataset-fixes-before-kg-build/results/manual_problems_ledger_measurements.json`. It supplies the machine-side numbers of [[database.manual-problems-ledger.2026.10]]; everything else in that ledger is quoted from an issue or PR body.

Read-only on every store and mirror; it writes only its results JSON. It counts LMDB entries of the 24 dev stores the ledger names, checks every bacterial `CITATION_KEY` (52 declared in `torchcell/datasets/ecoli` and `pputida`, plus 4 keys no loader declares: Rand 2017, Thompson 2019 lysine, Banerjee 2024, Nichols 2011) for a Zotero-backed `torchcell-library/` manifest, checks whether the four walled deposits are on disk, lists the dev-store manifests whose closure records no module-level binding (#734's residual), and with `--disk` and `--probe-pubchem` runs `du` over the retired dev-store copies and the two `/db` leftovers and sends six sequential PubChem PUG REST name lookups.

Run on 2026-10-10 at 23:43 and 23:45 CDT on `edd48664ae` (GilaHyper); the two runs agree, and the second added the manifest check:

| measurement | value |
|---|---|
| bacterial citation keys checked | 56 |
| keys with no Zotero-backed library mirror | 12 (9 raw-mirror only, plus Teteneva 2024, Banerjee 2024, Nichols 2011) |
| walled deposits present | 0 of 4 |
| `*.superseded.*` beside the dev stores | 508 G over 838 directories |
| `deprecated-*` beside the dev stores | 293 G over 69 directories |
| `/scratch/projects/torchcell-deprecated` | 68 G, 162 entries |
| `/db/database/biocypher-out/2026-10-09_07-20-33` | 170 G |
| `/db/database/data.superseded.2026-10-06_04-41-16` | 135 G |
| PubChem PUG REST from GilaHyper | 6 of 6 HTTP 200 |
| manifests with no binding closure node | 7: the four Costanzo dev subsets and three retired copies (#734's residual) |

Nichols 2011 reads as "no Zotero item" here only because the check needs a library manifest, which an item without a PDF never gets; the group item `REEL3RAD` exists (#691).
