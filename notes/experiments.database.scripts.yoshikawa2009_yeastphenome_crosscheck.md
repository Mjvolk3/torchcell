---
id: c50vwwd1n352b0a3qjvxejr
title: Yoshikawa2009_yeastphenome_crosscheck
desc: ''
updated: 1791802882860
created: 1791802882860
---

## 2026.10.12 - Yoshikawa 2009 against yp-data's curation of it

Issue #1000: yp-data v1.0 curates Yoshikawa 2009 (PMID 19054128) from the same Table S1 [[torchcell.datasets.scerevisiae.yoshikawa2009]] reads. Its `code.ipynb` averages the two culture columns of each condition per translated ORF and divides each stress by the nonstress mean. The script recomputes that from the store, reads both sides read-only and writes `experiments/database/results/yoshikawa2009_yeastphenome_crosscheck.json`; the three yp-data files are fetched by hand at commit `83e2917b` (the curl recipe is in the docstring) and verified against the sha256 pins in the script, and the notebook lines the recomputation follows are checked verbatim.

Measured on the dev store (slurm job 4166, build commit `f994bbbbe`, 28,140 records):

| comparison | joined cells | agree to 1e-12 | max abs difference | Pearson | Spearman |
|---|---|---|---|---|---|
| store vs `yoshikawa_shimizu_2009_value.txt` (8% ethanol, 1 M NaCl) | 9,348 over 4,674 genes | 9,302 | 0.0667 | 0.99977 | |
| store vs `yoshikawa_shimizu_2009_valuez.txt` (NPV) | 9,348 | | | 0.872 | 0.719 |

- The yeastphenome dev store holds 0 Yoshikawa rows (no `19054128_` file among its 47 `raw/*_valuez.txt`).
- yp-data carries the 5% ethanol column too ("EtOH (ethanol [5%])", 4,679 cells); the loader refuses it.
- The 46 cells that do not agree lie on 23 genes, and all 46 are reproduced to 1e-12 once the strains the loader refuses are pooled back into the gene yp-data translated them to: 22 merged ORFs into their locus (YDL038C into YDL039C, YAR044W into YAR042W, ...) and YDL134C-A into YDL133C-A. So yp-data's ORF translation averages a strain that deleted part of a locus with the strain that deleted the locus. The three other merged ORFs land on loci yp-data does not carry (YIL167W, YIR043C, YER109C).
- RPL22B's two strains (YFL034C-A, YFL035C-B) agree: the recomputation pools them per gene, as yp-data does.
- 30 store cells (15 genes) have no yp-data cell: the 13 non-gene loci the store keeps under their own ids (pseudogenes, blocked reading frames and the transposable-element gene YJR026W), YER087C-A and YFL013W-A.
