---
id: mxzux21dzfp6d95bdsi4gjg
title: Dechiara2022
desc: ''
updated: 1791716143052
created: 1791716143052
---

## 2026.10.11 - Release inventory and the growth-block misalignment

De Chiara et al. 2022 (Nat Ecol Evol 6:448-460, doi 10.1038/s41559-022-01671-9, mirror key `dechiaraDomesticationReprogrammedBudding2022`, Zotero group item N6R6CPDT) phenotyped the 1,011-isolate panel of Peter 2018 for three parts of the life cycle: asexual growth in 43 carbon and nitrogen environments, the sexual cycle (sporulation), and chronological life span (CLS). The DOI in the mirror manifest is `10.1038/s41559-022-01671-9`.

Everything per isolate is in one sheet, Supplementary Table 3 of `si/si4.xlsx` (MOESM4, sha256 `b2eb2e28cdd2c3037ce14bb1eb8f58ee84609157dd592b3904595b24daf8c99e`): 1,011 rows keyed by `Standard name`, 10 metadata columns, and 185 phenotype columns. Supplementary Table 10 names "the 99 phenotypes measured in this study": 5 sporulation, 8 CLS, 43 doubling times and 43 yields. Supplementary Table 3 carries each growth trait twice, once as the control-normalized log2 value the paper analysed and once as an absolute value (hours, cells), so 13 + 4 x 43 = 185.

Measured by [[experiments.036-dataset-fixes-before-kg-build.scripts.dechiara2022_release_inventory]] (`results/dechiara2022_release_inventory.json`):

| quantity | value |
|---|---|
| isolates in Supplementary Table 3 | 1,011, names unique |
| isolate names found in Peter's presence/absence matrix | 1,011 of 1,011 |
| phenotype columns whose non-missing count differs from Supplementary Table 10 | 0 |
| `SACE_` isolates with any growth value | 0 of 93 |
| rows whose normalized and absolute growth blocks describe the same isolate | 386 |
| rows whose two blocks describe different isolates | 519 |
| rows with no absolute growth value / too few values to match | 105 / 1 |

**The two growth blocks disagree on 519 rows.** Where the blocks agree, `log2(absolute) - normalized` is a per-environment constant (largest within-environment SD 0.032 log2 units for doubling time, 0.28 for yield), which is the back-transform the authors describe in the peer-review file (`si/si3.md`: "We now also show and report in table S3 the absolute cell doubling time and cell yield, in hours and numbers of cells"). Matching every row's 86-value profile against every row of the other block finds a best partner with Pearson r >= 0.964 for 99% of rows, and the partner of 516 rows is the alphabetically PREVIOUS isolate name (`BDM` through `CRL`): the normalized block is shifted by one row over a contiguous alphabetical run. `BDL` has normalized values but no absolute ones and `CRL` the reverse, the two ends of the run.

**Genotype adjudication says the absolute block is the correctly named one.** In the 386 agreeing rows both blocks carry the textbook sugar associations with Peter 2018's pangenome presence (two-sided Mann-Whitney): SUC2 (`YIL162W`) and sucrose 2% doubling time p = 3.1e-4 / 3.0e-4 (normalized / absolute); MAL13 (`YGR288W`) and maltose 32% yield p = 4.1e-10 / 3.3e-10. In the 519 disagreeing rows only the absolute block keeps them: sucrose 2% p = 0.68 (normalized) against 3.1e-3 (absolute); maltose 32% yield p = 0.30 against 6.0e-14; maltose 32% doubling time p = 0.42 against 2.5e-8.

**The life-cycle columns are correctly named.** Recomputing Extended Data Fig. 7b from the released CLS columns and Peter's pangenome ORF `X560` (HPF1-like) gives n = 89 vs 718 and p = 1.9e-9, 8.6e-9, 1.6e-5 for crSDC days 7, 21, 35, the caption's values exactly. The sporulation and CLS columns also equal, name for name, the per-isolate source-data workbooks for Extended Data Figs. 3b, 4a, 5 and 6.

Decision taken by recommendation: the loader reads the ABSOLUTE growth block only and refuses the normalized block, whose row names are wrong for 519 isolates and which carries no information beyond a per-environment constant where it is right. The misalignment is reported upstream as a finding, not repaired by us (repairing it would mean re-assigning released numbers to names by our own inference).
