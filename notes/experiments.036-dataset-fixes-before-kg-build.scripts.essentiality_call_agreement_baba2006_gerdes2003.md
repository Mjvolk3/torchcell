---
id: l65z5ynsjsvvj37gfo2fwvo
title: Essentiality_call_agreement_baba2006_gerdes2003
desc: ''
updated: 1791695676730
created: 1791695676730
---

## 2026.10.11 - G4 by content: Baba 2006 and Gerdes 2003 against the served essentiality stores

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/essentiality_call_agreement_baba2006_gerdes2003.py`. Results: `experiments/036-dataset-fixes-before-kg-build/results/essentiality_call_agreement_baba2006_gerdes2003.json`. Reads the four dev-tree LMDBs read-only. The candidate gate's own `overlap` refuses all four, because each manifest records a dirty tree (an untracked guard file counts), so this committed script is the comparison of record.

Stores: Baba and Gerdes built at `ee131b44`; Goodall 2018 at `529b6bf5`; Price 2018 Table S1 at `52d376d6` (all `torchcell_dirty: true`). Gerdes is carried from MG1655 to BW25113 by the one-to-one ECK pairs (3,578 of 3,618 loci; 40 have no pair).

| pair (shared BW25113 loci) | shared | both E | both not | first only E | second only E | agree | kappa |
|---|---|---|---|---|---|---|---|
| Baba vs Goodall TL | 3,961 | 259 | 3,588 | 26 | 88 | 3,847 | 0.80 |
| Baba vs Goodall LB | 3,969 | 253 | 3,592 | 27 | 97 | 3,845 | 0.79 |
| Gerdes vs Goodall TL | 3,465 | 243 | 2,870 | 311 | 41 | 3,113 | 0.53 |
| Gerdes vs Goodall LB | 3,468 | 238 | 2,867 | 310 | 53 | 3,105 | 0.51 |
| Baba vs Gerdes | 3,551 | 205 | 2,910 | 47 | 389 | 3,115 | 0.43 |

Price 2018 Table S1 is essential-only (320 loci), so it is compared as a list: Baba shares 319 and calls 235 of them essential; Gerdes shares 274 and calls 218 essential.

The Baba vs Gerdes "both essential" count, 205, is the number Baba 2006 states ("only 67% (205 genes) overlap with the predicted essential genes in this study"), which checks both identifier routes end to end.

Verdict: neither row is subsumed. No served store holds the same call on every shared locus, so both are independent datasets and both load in full.
