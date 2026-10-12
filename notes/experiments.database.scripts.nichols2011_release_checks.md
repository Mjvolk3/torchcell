---
id: kyzc7mlo19pdaalqr8xtdq7
title: Nichols2011_release_checks
desc: ''
updated: 1791817761720
created: 1791817761720
---

## 2026.10.12 - Measurements on the Nichols 2011 release (#691)

`experiments/database/scripts/nichols2011_release_checks.py` reads the deposit and Shiver 2016's S1 Dataset read-only (no genome annotation) and writes `experiments/database/results/nichols2011_release_checks.json` (gitignored; rerun to regenerate).

- **G3 inventory.** 8 deposited files, all intact against the manifest. `mmc1.xls` BIFF, Table S1, 114 screen rows, Screens sum 324 = its Total Screens cell. `mmc2.xls` OOXML under an `.xls` name, Table S2, 3,979 x 324, 1,286,962 numbers and 2,234 empty cells, -44.33 to 9.30. `mmc3`-`mmc7` BIFF analysis tables (94 strain labels; 116 + 102 labels; 831 lethal calls; 832 gene pairs; 719 drug-GO rows). `mmc8.pdf` the article and its Supplemental Information.
- **Standardization.** Per-column median -0.019 to 0.022, IQR 1.309 to 1.379 (median 1.348).
- **Twelve labels on two rows**, zero identical co-present scores in each pair, profile r -0.16 to 0.55.
- **Shiver 2016 batch 0.** 235 conditions matched one to one onto Table S2 columns by profile correlation (best r 0.27 to 0.93, median 0.80); strains joined by released symbol (3,470 in both); 778,854 of 888,431 cells overlap a stored (strain, column); 0 identical, 9,410 within 0.01, pooled r 0.800, median |d| 0.458.
