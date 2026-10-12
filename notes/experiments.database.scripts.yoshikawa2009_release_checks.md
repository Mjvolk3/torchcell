---
id: 5kk9qjoyv31gfremkp5sblw
title: Yoshikawa2009_release_checks
desc: ''
updated: 1791802874756
created: 1791802874756
---

## 2026.10.12 - The source question and the release, measured

`experiments/database/scripts/yoshikawa2009_release_checks.py --check-source` reads the two deposited copies of Table S1, the paper OCR and the R64-4-1 genome, and writes `experiments/database/results/yoshikawa2009_release_checks.json`. Loader: [[torchcell.datasets.scerevisiae.yoshikawa2009]].

### Publisher `.xls` against YeastPhenome `.xlsx`

| check | `.xls` (publisher) | `.xlsx` (yp-data v1.0) |
|---|---|---|
| sha256 | `1861ba64...f759` | `f8567d2c...3871` |
| sheets | `data` | `data` |
| dimensions | 4,731 x 14 | 4,731 x 14 |
| merged header ranges | C1:E1, F1:H1, I1:K1, L1:N1 | the same four |
| condition headers | non-stress, 5% ethanol, 8% ethanol, 1M NaCl | the same |
| author / last saved by | katsunori yoshikawa / Katsunori Yoshikawa2, 2008-09-19 12:29:08 | katsunori yoshikawa / Anastasia Baryshnikova, 2013-03-06 18:09:55 |

All 66,234 cells compared by value and kind: 9,476 text, 56,748 numeric and 10 empty in both, **0 differences**, no formulas. The publisher's own file carries the 5% ethanol columns; the curator did not add them. The `.xls` was last saved on the paper's revision date ("revised 19 September 2008"); its BIFF WRITEACCESS user is "manjula". The yp-data copy re-fetched by its recorded retriever (`direct_url`, commit `83e2917b`) produced the deposited sha256.

### The 5% ethanol condition

The paper's OCR writes no 5 percent anywhere (regex over `paper.md`: 0 spans; 8% appears 26 times) and its Methods describe only nonstress, 8% ethanol and 1 M NaCl; Table 1 gives the standard strain's rate and thresholds for those three only. Table S1's 5% calls are 893 sensitive, 3,728 no change, 108 tolerant. Measured: they are consistent with a both-cultures threshold between 0.2931 and 0.2932 (sensitive) and between 0.3105 and 0.3106 (tolerant). Hypothesis (untested): under the paper's mean +/- 3.3 SD rule that implies an unreleased standard-strain mean near 0.302 and SD near 0.0026; this is arithmetic on the bracket, not a sourced reference value, so the loader refuses the condition.

### Calls

Table S1's calls reproduce the paper's Table 2 exactly (nonstress 591 / 4,095 / 43; 8% ethanol 864 / 3,862 / 3; 1 M NaCl 637 / 4,080 / 12). The both-cultures rule ("growth rates in both independent cultures were lower than the threshold value") with Table 1's printed thresholds reproduces all but 6, 12 and 10 calls, in each the deciding culture lies within 0.0004 of the printed threshold (measured), which fits thresholds printed to three decimals. Hypothesis (untested): the authors applied the unrounded mean +/- 3.3 SD.

### Names

Column B against Column A under R64-4-1 candidate sets: 1,456 equal, 3,197 whose unique candidate is Column A, 70 where Column A is one of several candidates, 6 with no candidate (pseudogenes and blocked reading frames, which carry no `gene=`), 0 that exclude Column A. Column B repeats one name (IMP2, for YMR035W and YIL154C, which is IMP2'). Column A writes two suffixes in lower case (YAL043C-a, YAR043C-a). The loader keeps 4,690 strains (21 renamed), refuses 39, and keeps RPL22B's two released strains (YFL034C-A, YFL035C-B) apart.
