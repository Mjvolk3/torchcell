---
id: gipn08qtlar8p1ec9p30ei2
title: Nichols2011
desc: ''
updated: 1791817636069
created: 1791817636069
---

## 2026.10.12 - Table S2 loaded as released (#691)

`EnvChemgenNichols2011Dataset` ([[torchcell.datasets.ecoli.nichols2011]], `torchcell/datasets/ecoli/nichols2011.py`) loads Nichols et al. 2011 (Cell 144:143, doi:10.1016/j.cell.2010.11.052) from the owner's deposit of the Cell supplemental information (`$DATA_ROOT/torchcell-raw/nicholsPhenotypicLandscapeBacterial2011/si/`, PR #1046). Owner rule, 2026-10-12: the data is the data; ingestion consumes the raw data and the genome never gates a record.

### What is stored

| quantity | value | source |
|---|---|---|
| Table S2 cells (3,979 strain rows x 324 screens) | 1,289,196 | `si/mmc2.xls` |
| records (non-empty cells) | 1,286,962 | `preprocess/cell_accounting.json` |
| empty cells, left absent as released | 2,234 | same |
| rows by released class | deletion 3,837 (5 multi-gene), affinity_tag (SPA) 117, sequence_allele 9, linked_marker 9, degron (DAS) 5, truncation 2 | `preprocess/strain_labels.json` |
| distinct ECK accessions | 3,949 | same |
| rows carrying `replicate_id` (12 labels on two rows) | 24 | same |
| genome lookups | 0 | same |

- **Strains.** Each row label's ECK accession(s) on `BacterialReleasedStrainPerturbation` (added in this PR, [[torchcell.datamodels.bacterial-perturbation-ontology]]) with the released symbol, the whole label (`strain_label`) and the class its suffix states (`released_kind`); `cassette="kanamycin"` ("All mutant strains are marked with or linked to kanamycin."). The release labels strains by ECK accession and symbol, not by JW number. A multi-gene label (`ECK1205/1207/1209-RDLABC`) names only the set, so each gene takes the name the release gives its accession on its own row (`ECK4143-ECNB`), else the accession; no accession is released under two names.
- **Scores.** The per-screen re-scaled S-score as released: `measurement_type=z_score`, `assay_type=colony_size_array`, reference 0. Measured on Table S2: every column's median lies within 0.022 of 0 and its IQR in 1.309 to 1.379 (median 1.348), the paper's "re-scaled the data to a standard normal distribution (IQR=1.35)". Range -44.33 to 9.30.
- **Replicates.** "four to six independent measurements": `n_samples=4` (conservative lower end, no per-record count and nothing to back-solve), `sample_unit=colony`; uncertainty and SE typed gaps.
- **Conditions.** Table S1 has 114 screen rows (the #1046 note's "115" counted the header) summing to its Total Screens cell, 324; the #1046 note's "648" summed the total row as well. Every row is transcribed verbatim and checked against the bytes; every Table S2 label maps one to one onto a Table S1 column. Compounds keep Table S1's names; 46 of 105 have no InChIKey in the curated table and carry the typed gap (#1063). `N-acetyl Glucosamine` is the one carbon source Table S1 does not mark "(M9)", so it sits on the default LB agar. `duration_hours` is a typed gap everywhere ("14-16 hrs" for the majority; "Several stresses ... required longer incubation times").
- **Article quotes** come from the MinerU OCR of `si/mmc8.pdf` (slurm 4300, CPU, 5 min 53 s, 4.3 GB; `si/mmc8.md` sha256 `0fbf3247...`), pinned in the raw manifest by `record_article_ocr`.

### Tables S3 to S7

Not loaded; each is an analysis of Table S2. mmc3: 94 multi-stress responsive strain labels. mmc4: 116 rich-media conditionally essential and 102 auxotroph labels. mmc5: 831 rich-media lethal (strain, condition) calls. mmc6: 832 orphan to annotated gene pairs, r >= 0.5. mmc7: 719 drug-GO scores (sheet `TableS5-Drug-GO interactions.tx`).

### G4 against Shiver 2016

Measured by [[experiments.database.scripts.nichols2011_release_checks]], release against release (strains by released symbol, conditions by profile correlation, no annotation read): Shiver's 235 batch-0 conditions match 235 distinct Table S2 columns; 778,854 of its 888,431 batch-0 cells land on a (strain, column) stored here; 0 identical values, 9,410 within 0.01, pooled Pearson r 0.800, median |difference| 0.458. A re-scoring, not a copy; `shiver2016.py` does not load batch 0, so nothing is served twice.

### Open owner questions

1. The 100 sRNA/small protein strains were "constructed in MG1655 (Hobbs et al., 2010) and used in that background"; Table S2 does not mark which rows they are, so every record pins BW25113.
2. Multi-gene labels: keep the per-accession released name, or store the set name (`RDLABC`) on every gene?
