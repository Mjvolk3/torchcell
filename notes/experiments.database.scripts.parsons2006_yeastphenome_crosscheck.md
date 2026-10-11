---
id: eapxd4125r2d5iyso6njmko
title: Parsons2006_yeastphenome_crosscheck
desc: ''
updated: 1791746756253
created: 1791746756253
---

## 2026.10.11 - Parsons 2006 against yp-data's curation of it

Issue #1000: yp-data v1.0 curates Parsons 2006 (PMID 16901791) from the same Table S7 [[torchcell.datasets.scerevisiae.parsons2006]] reads. The script reads both sides read-only and writes `experiments/database/results/parsons2006_yeastphenome_crosscheck.json`. The two yp-data files are fetched by hand at commit `83e2917b` (the curl recipe is in the docstring) and verified against the sha256 pins in the script.

Measured on the clean dev store (build commit `93f986695`, 291,673 records):

| comparison | joined cells | Pearson | Spearman | exact negations |
|---|---|---|---|---|
| store vs `parsons_boone_2006_value.txt` | 291,530 | -0.999993 | -0.999976 | 291,457 |
| store vs `parsons_boone_2006_valuez.txt` (NPV) | 291,530 | -0.917 | -0.974 | 0 |

- The yeastphenome dev store holds 0 Parsons rows (no `16901791_` file among its 47 `raw/*_valuez.txt`).
- yp-data's curated value is the negated release, -log2(control/treated). The 73 joined cells that are not a negation are all YDL039C: the release carries YDL039C and YDL038C (which R64 merges into YDL039C), yp-data carries no YDL038C row and its YDL039C row is not the release's YDL039C row.
- 143 store cells have no yp-data cell (YDL038C, and YGR272C, which yp-data lists as YGR271C-A).
