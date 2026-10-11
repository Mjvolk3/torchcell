---
id: zauksxesxo9rxmn6llq538f
title: Dechiara2022_release_inventory
desc: ''
updated: 1791716174743
created: 1791716174743
---

## 2026.10.11 - What the script measures

Inventories De Chiara 2022's release (`si/si4.xlsx`, MOESM4, sha256 `b2eb2e28cdd2c3037ce14bb1eb8f58ee84609157dd592b3904595b24daf8c99e`) and writes `experiments/036-dataset-fixes-before-kg-build/results/dechiara2022_release_inventory.json`:

- every sheet's shape, the non-missing count of every Supplementary Table 3 phenotype column against Supplementary Table 10 (0 of 185 differ);
- the alignment of the normalized and absolute growth blocks, row profile against row profile (386 rows agree, 519 do not, 105 carry no absolute value, 1 too few to match);
- the per-environment `log2(absolute) - normalized` offset over the agreeing rows;
- a genotype adjudication of the disagreeing rows with Peter 2018's pangenome presence matrix (SUC2, MAL11, MAL13 against sucrose and maltose growth), which keeps the absolute block;
- Extended Data Fig. 7b recomputed from the released CLS columns (HPF1-like, pangenome `X560`), which matches the caption.

Findings and the loader decision they support: [[torchcell.datasets.scerevisiae.dechiara2022]].
