---
id: 6rvrmynzlz1n2rq1o75zpra
title: Balakrishnan2022_release_inventory
desc: ''
updated: 1791613578751
created: 1791613578751
---

## 2026.10.10 - Settling schedule row 57, Balakrishnan 2022 multi-omics

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/balakrishnan2022_release_inventory.py`
Results: `results/balakrishnan2022_release_inventory.json`

Balakrishnan, Mori, Segota, Zhang, Aebersold, Ludwig and Hwa 2022, *Science* 378 eabk2066,
doi `10.1126/science.abk2066`, PMC9804519 (NIH author manuscript NIHMS1856867). Verdict:
**no loader on this pass.** One arm is scriptable and needs a schema decision plus two
medium objects; every other arm is behind a wall. Issues #853 (by-hand mirror) and #854
(schema).

### Arms, as released

The schedule row describes paired transcriptome and proteome plus degradation rates. There
is no metabolic flux arm in this paper; "mRNA synthesis flux" is a total transcription
rate, not a reaction flux.

| arm | file | scriptable | measured content |
|---|---|---|---|
| steady-state transcriptome | Table S3 sheet 2 (GEO `GSE205717`) | yes, sha256 `9d7df03b...71cc20` | 4,342 genes x 29 columns of mRNA number fractions; columns sum to 1 within 2.3e-8 |
| rifampicin decay time courses | Table S3 sheet 4 | yes | 4,342 genes x 16 columns (2 series x 8 time points, 0 to 11 min) |
| proteome | Table S4 | no (#853) | not measured |
| translation-efficiency fold changes | Table S5 | no | not measured |
| degradation rates, promoter on-rates, initiation rates | Table S6 | no | not measured (~2,700 genes per the main text) |
| unnamed in text | Table S7 | no | not measured |
| strains, conditions, methods | SI PDF (Tables S1, S2) | no | not measured |

Access, measured 2026-10-10: the PMC bucket holds only `.json/.txt/.xml`; the PMC `bin/`
URL answers the proof-of-work page; Europe PMC says "not open access"; `science.org` and
`europepmc.org/.../bin/` answer 403. Only GEO serves Table S3.

### Table S3, measured

- 29 described samples: C-limitation 10, A-limitation 8, R-limitation 11. Strains NCM3722
  and its derivatives NQ1243 / NQ1390 (Pu-ptsG) and NQ393 (Plac-GOGAT). Media M9 with
  11.34 mM (NH4)2SO4 (C, A series) and MOPS with 10 mM NH4Cl (R series). Growth 0.24 to
  0.95 per hour.
- `a4_1` heads two columns (16 and 17) identical in 4,342 of 4,342 rows; the described
  sample `a3_1` has no column. 27 distinct loadable columns.
- Three nitrogen cells read "11.34 mM (NH4)2SO5", "SO6", "SO7" (rows a2_1, a3_1, a4_1),
  verbatim; every neighboring row reads "(NH4)2SO4".
- Identifiers: 4,324 b-numbers plus 18 rows with locus `0`; 4,176 of 4,325 distinct loci
  are in the MG1655 gene set. Why the 148 b-numbers miss is not measured.
- No integer read counts are released, which is what blocks `RNASeqExpressionPhenotype`
  (#854).

### Duplication against Mori 2021 and Schmidt 2016

- Table S3 lists the same 4,342 gene names in the same order as Mori 2021 Dataset EV9.
- Its strain set equals Mori 2021 EV3's 29 C-, A- and R-limitation rows, the same base
  media and nitrogen sources. Two samples (`c5`, `c2`) match a Mori EV3 row on strain,
  medium, supplement and growth rate. Hypothesis (untested): those are shared cultures.
- The proteome arm (Table S4) cannot be compared: it is behind #853. The data statement
  says the proteomic files "have been published previously(10)" (ref 10 = Mori 2021,
  PXD014948). Hypothesis (untested): Table S4 re-serves Mori EV9, in which case it is
  subsumed and only the transcriptome and rate arms are new. Mori's C/A/R series are
  themselves not on main (dropped by `DROP_NO_MEDIA_ENTRY`); the served Mori store holds
  only the 7 MG1655 calibration samples. Overlap with Schmidt 2016 is not measured; it is a
  proteome, so it could only bear on Table S4.
- Cross-layer r (log10 Pearson, keys positive in both), mRNA fraction vs Mori protein mass
  fraction, a layer comparison and not a duplication test:

| pair | growth (/h) | shared keys | Pearson log10 | Spearman |
|---|---|---|---|---|
| `c5` (mRNA, M9 ref) vs Mori `C2` (protein, M9 ref) | 0.91 / 0.91 | 1,917 | 0.780 | 0.773 |
| `r0` (mRNA, MOPS ref) vs Mori `A2` (protein, MOPS ref) | 0.89 / 0.98 | 1,938 | 0.725 | 0.722 |
| replicate `c5` vs `c0_1` (mRNA) | | 4,218 | 0.990 | 0.992 |
| replicate `r0` vs `r0_1` (mRNA) | | 4,219 | 0.989 | 0.991 |
| replicate Mori `A2` vs `H1` (protein) | | 1,885 | 0.992 | 0.992 |

Verdict: the transcriptome arm is independent (r 0.73 to 0.78 against the sister proteome,
against 0.99 within each layer), so it is not a re-served proteome.

### What unblocks a loader

1. #853: the SI and Tables S4-S7 deposited by hand.
2. #854: a schema decision for a count-less transcriptome (recommended: optional
   `expression_count` with a number-fraction `measurement_type`, which rides KG 4.0).
3. Two medium objects (M9 with 11.34 mM (NH4)2SO4; MOPS Record's), sourced from the SI;
   the same two objects would admit Mori 2021's 29 dropped EV9 samples.
4. Strains NQ1243 / NQ1390 / NQ393 on the MG1655 pin, following Gupta 2024's NCM3722
   precedent.

The loader would then follow the Ishii 2007 / Caglar 2017 pattern: one module per arm
(steady-state transcriptome, Table S6 rates, Table S4 only if it is not Mori EV9) over a
shared sourcing layer.
