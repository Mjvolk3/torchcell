---
id: 2978587zhvd0f35fzag0pjv
title: Yeast Datasets Table
desc: ''
updated: 1791956706147
created: 1791956706147
---

## Yeast datasets: Instances vs Signal (gzip), with the Signal controls

<!-- GENERATED: raw data from experiments/database/scripts/build_bacteria_datasets_table.py
(-> results/pre-build/<date>/<organisms>_datasets.json); this note is a VIEW rendered by
render_bacteria_datasets_table.py. Do not edit tables by hand. -->

Pre-build inventory of the S. cerevisiae datasets built as LMDBs in the dev tree
(the set `build_adapter_map(organisms={scerevisiae})` serves), a temporary counterpart of
[[paper.supported-datasets-and-databases]] (issue #1098), with the Signal-column controls of
issue #1099 measured in the same pass. Commentary and findings:
[[experiments.database.scripts.build_bacteria_datasets_table]].

**Columns.** Every column is read from the built LMDB; nothing is hand-typed. *Genotypes* =
distinct stored genotype payloads; *Env* = distinct stored environment payloads; *Instances* =
record count; *Phenotype* = the stored `label_name` (and `measurement_type` when set); *Shape* =
shape of one phenotype instance; *Graph role* = where the label sits in the cell graph;
*Signal (gzip, bits)* = gzip size, in bits, of the concatenated stored instances (perturbation +
environment + phenotype, sorted keys), the yeast table's definition.

**Controls** (second table). *Structure* = the same serialization with every leaf replaced by a
type token (keys, nesting and list lengths kept). *Shuffled* = numeric leaves permuted within
their key path across the dataset (marginals kept, joint structure destroyed). *Noise* = every
digit of every numeric leaf redrawn uniformly at the stored precision (the ceiling for that
structure). *Binary* = numeric leaves z-scored per key path and thresholded at 0, written as
`1`/`0` in place. *Packed 1-bit* = gzip of just those bits, 8 per byte, no structure (the one-bit
floor on value-carried information). Ratios are dimensionless.

**Caveat.** Several bacterial stores intern the environment as a `$ref` hash plus a name rather
than the full media payload, so their environment bytes are a 64-character digest; the yeast
2026-07-17 snapshot predates interning. Signal is a relative proxy, comparable within this
table, not an absolute information content.

### Datasets

#### Environmental / chemogenomic response

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| CellYieldDeChiara2022 | S. cerevisiae | 906 | 43 | 36,286 | environment_response (cell_yield) | scalar | global | 1.7×10⁷ |
| CrisprMagicLian2019 | S. cerevisiae | 266,304 | 3 | 266,304 | environment_response (log2_ratio) | scalar | global | 1.3×10⁸ |
| CrispriChemgenSmith2016 | S. cerevisiae | 1,035 | 26 | 14,463 | environment_response (log2_ratio) | scalar | global | 7.5×10⁶ |
| CrispriMormino2022 | S. cerevisiae | 12 | 1 | 12 | environment_response (categorical) | scalar | global | 1.4×10⁴ |
| DoublingTimeDeChiara2022 | S. cerevisiae | 906 | 43 | 36,295 | environment_response (growth_rate) | scalar | global | 1.7×10⁷ |
| EnvChemgenAuesukaree2009 | S. cerevisiae | 333 | 6 | 525 | environment_response (categorical) | scalar | global | 3.1×10⁵ |
| EnvChemgenCostanzo2021 | S. cerevisiae | 4,422 | 14 | 61,640 | environment_response (differential_fitness) | scalar | global | 1.1×10⁸ |
| EnvChemgenHoepfner2014 | S. cerevisiae | 10,839 | 5,586 | 30,308,574 | environment_response (sensitivity_score) | scalar | global | 2.1×10¹⁰ |
| EnvChemgenMota2024 | S. cerevisiae | 607 | 3 | 1,289 | environment_response (ordinal) | scalar | global | 7.4×10⁵ |
| EnvChemgenParsons2006 | S. cerevisiae | 4,111 | 82 | 329,254 | environment_response (log2_ratio) | scalar | global | 2.2×10⁸ |
| EnvChemgenPiotrowski2017 | S. cerevisiae | 289 | 11,949 | 3,435,820 | environment_response (z_score) | scalar | global | 1.4×10⁹ |
| EnvChemgenVanacloig2022 | S. cerevisiae | 3,649 | 45 | 164,205 | environment_response (log2_ratio) | scalar | global | 9.3×10⁷ |
| EnvChemgenWildenhain2015 | S. cerevisiae | 258 | 5,178 | 449,033 | environment_response (z_score) | scalar | global | 2.6×10⁸ |
| EnvChemgenYoshikawa2009 | S. cerevisiae | 4,729 | 4 | 37,832 | environment_response (growth_rate) | scalar | global | 1.0×10⁷ |
| FattyAcidSmith2006 | S. cerevisiae | 4,770 | 3 | 14,310 | environment_response (ordinal) | scalar | global | 1.8×10⁶ |
| HetHillenmeyer2008 | S. cerevisiae | 17,659 | 512 | 3,074,132 | environment_response (log2_ratio) | scalar | global | 2.0×10⁹ |
| HomHillenmeyer2008 | S. cerevisiae | 9,220 | 276 | 1,228,676 | environment_response (z_score) | scalar | global | 8.4×10⁸ |
| IsolateGrowthPeter2018 | S. cerevisiae | 971 | 35 | 33,985 | environment_response (condition_growth_ratio) | scalar | global | 1.6×10⁷ |

#### Expression (RNA-seq)

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Boocock2025 | S. cerevisiae | 79,516 | 2 | 79,516 | umi_count (scrnaseq_umi_count) | vector (3797) | node | 8.9×10⁹ |
| CaudalPanTranscriptome2024 | S. cerevisiae | 943 | 1 | 943 | expression_tpm (rnaseq_tpm) | vector (6017) | node | 1.7×10⁹ |
| NadalRibellesPerturbSeq2025 | S. cerevisiae | 3,150 | 2 | 6,188 | expression_log2_ratio (pseudobulk_scrnaseq_log2fc) | vector (5818) | node | 2.5×10⁹ |
| Nguessan2025 | S. cerevisiae | 2,951 | 1 | 2,951 | umi_count (scrnaseq_umi_count) | vector (1670) | node | 4.5×10⁸ |
| PooledPromoterExpressionVaishnav2022 | S. cerevisiae | 71,158 | 1 | 71,158 | expression_level | scalar | node | 1.3×10⁸ |
| TfDeletionPseudobulkJackson2020 | S. cerevisiae | 72 | 11 | 784 | umi_count (scrnaseq_umi_count) | vector (5075) | node | 9.7×10⁷ |

#### Expression (microarray)

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| DmMicroarraySameith2015 | S. cerevisiae | 72 | 1 | 72 | expression_log2_ratio | vector (6169) | node | 1.7×10⁸ |
| InductionTimecourseHackett2020 | S. cerevisiae | 214 | 31 | 1,752 | log2_cleaned | vector (6175) | node | 4.7×10⁹ |
| MicroarrayKemmeren2014 | S. cerevisiae | 1,484 | 1 | 1,484 | expression_log2_ratio | vector (6169) | node | 3.2×10⁹ |
| MicroarrayScoreHu2007 | S. cerevisiae | 263 | 1 | 263 | expression_score (error_model_x_score_weighted_duplicate_average) | vector (6371) | node | 2.5×10⁸ |
| SmMicroarraySameith2015 | S. cerevisiae | 82 | 1 | 82 | expression_log2_ratio | vector (6169) | node | 1.8×10⁸ |

#### Fitness

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Bloom2019 | S. cerevisiae | 13,950 | 38 | 530,100 | environment_response (control_regression_residual) | scalar | global | 2.6×10⁸ |
| CrispriSegregantHale2024 | S. cerevisiae | 895,494 | 2 | 8,692,635 | environment_response (relative_fitness) | scalar | global | 6.7×10⁹ |
| DmfCostanzo2016 | S. cerevisiae | 20,705,612 | 2 | 20,705,612 | fitness | scalar | global | 7.1×10⁹ |
| DmfKuzmin2018 | S. cerevisiae | 410,571 | 1 | 410,571 | fitness | scalar | global | 1.5×10⁸ |
| DmfKuzmin2020 | S. cerevisiae | 608,805 | 1 | 632,998 | fitness | scalar | global | 2.4×10⁸ |
| SmfBaryshnikova2010 | S. cerevisiae | 6,023 | 2 | 6,023 | fitness | scalar | global | 1.8×10⁶ |
| SmfCostanzo2016 | S. cerevisiae | 10,277 | 2 | 20,484 | fitness | scalar | global | 5.4×10⁶ |
| SmfKuzmin2018 | S. cerevisiae | 1,539 | 1 | 1,539 | fitness | scalar | global | 3.9×10⁵ |
| SmfKuzmin2020 | S. cerevisiae | 472 | 1 | 472 | fitness | scalar | global | 9.3×10⁴ |
| SmfODuibhir2014 | S. cerevisiae | 1,312 | 1 | 1,312 | fitness | scalar | global | 2.0×10⁶ |
| TmfKuzmin2018 | S. cerevisiae | 91,111 | 1 | 91,111 | fitness | scalar | global | 3.9×10⁷ |
| TmfKuzmin2020 | S. cerevisiae | 290,355 | 1 | 301,798 | fitness | scalar | global | 1.3×10⁸ |

#### Gene interaction

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| DmiCostanzo2016 | S. cerevisiae | 20,705,612 | 2 | 20,705,612 | gene_interaction | scalar | edge | 6.0×10⁹ |
| DmiKuzmin2018 | S. cerevisiae | 410,399 | 1 | 410,399 | gene_interaction | scalar | edge | 1.3×10⁸ |
| DmiKuzmin2020 | S. cerevisiae | 608,604 | 1 | 632,797 | gene_interaction | scalar | edge | 2.1×10⁸ |
| TmiKuzmin2018 | S. cerevisiae | 91,111 | 1 | 91,111 | gene_interaction | scalar | hyperedge | 3.4×10⁷ |
| TmiKuzmin2020 | S. cerevisiae | 290,355 | 1 | 301,798 | gene_interaction | scalar | hyperedge | 1.1×10⁸ |

#### Genome architecture

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| GenomeArchitecturePuddu2019 | S. cerevisiae | 4,550 | 1 | 8,835 | genome_architecture | vector (28) | global | 1.2×10⁷ |

#### Life cycle

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| LifeCycleDeChiara2022 | S. cerevisiae | 905 | 5 | 4,039 | life_cycle | vector (4) | global | 1.3×10⁶ |

#### Metabolite

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| AminoAcidCooper2010 | S. cerevisiae | 4,334 | 1 | 4,382 | metabolite_level (ce_lif_peak_area_ratio_to_plate_mean) | vector (16) | bipartite node | 5.2×10⁶ |
| AminoAcidMulleder2016 | S. cerevisiae | 4,678 | 1 | 4,678 | metabolite_level (intracellular_concentration_mM) | vector (19) | bipartite node | 1.1×10⁷ |
| BetaxanthinCachera2023 | S. cerevisiae | 4,759 | 1 | 4,759 | metabolite_level (cri_spa_corrected_hsv_yellowness_24h) | scalar | bipartite node | 4.5×10⁶ |
| FattyAcidXue2025 | S. cerevisiae | 176 | 1 | 176 | metabolite_level (titer_mg_per_l) | vector (5) | bipartite node | 1.9×10⁵ |
| IsobutanolScreenLopez2024 | S. cerevisiae | 4,607 | 1 | 4,805 | metabolite_level (biosensor_gfp_fluorescence_fold_change) | scalar | bipartite node | 9.0×10⁵ |
| IsobutanolValidatedLopez2024 | S. cerevisiae | 226 | 1 | 227 | metabolite_level (biosensor_gfp_fluorescence_fold_change) | scalar | bipartite node | 7.4×10⁴ |
| MetaboliteDaSilveira2014 | S. cerevisiae | 127 | 1 | 127 | metabolite_level (lipidomics_ms_relative_abundance_au) | vector (135) | bipartite node | 1.1×10⁶ |
| MetaboliteZelezniak2018 | S. cerevisiae | 95 | 1 | 129 | metabolite_level (lc_srm_signal_batch_corrected_uncalibrated_dataset1) | vector (14) | bipartite node | 3.5×10⁵ |
| OrganicAcidYoshida2012 | S. cerevisiae | 17 | 1 | 17 | metabolite_level (hplc_organic_acid_titer_mM) | vector (6) | bipartite node | 3.3×10⁴ |

#### Morphology

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| CarotenoidOzaydin2013 | S. cerevisiae | 4,475 | 1 | 4,607 | visual_score | scalar | global | 7.7×10⁶ |
| ScmdOhnuki2018 | S. cerevisiae | 1,112 | 1 | 1,112 | calmorph | vector (281) | global | 4.9×10⁷ |
| ScmdOhnuki2022 | S. cerevisiae | 1,982 | 1 | 1,982 | calmorph | vector (281) | global | 8.8×10⁷ |
| ScmdOhya2005 | S. cerevisiae | 4,718 | 1 | 4,718 | calmorph | vector (281) | global | 1.5×10⁸ |

#### Protein abundance

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ProteomeMessner2021 | S. cerevisiae | 1 | 18 | 70 | protein_abundance (swath_ms_diann_maxlfq_proteotypic_quantity) | vector (1920) | node | 1.2×10⁷ |
| ProteomeMessner2023 | S. cerevisiae | 4,549 | 1 | 4,699 | protein_abundance (swath_ms_maxlfq_batch_corrected_quantity) | vector (1830) | node | 1.2×10⁹ |
| ProteomeMuenzner2024 | S. cerevisiae | 795 | 1 | 795 | protein_abundance (swath_ms_maxlfq_precursor_batch_corrected_quantity) | vector (1542) | node | 9.6×10⁸ |
| ProteomeZelezniak2018 | S. cerevisiae | 97 | 1 | 97 | protein_abundance (swath_ms_label_free_log_signal_sva) | vector (726) | node | 1.7×10⁷ |

#### Protein synthesis / turnover

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ProteinTurnoverMuenzner2024 | S. cerevisiae | 55 | 1 | 55 | degradation_rate (dynamic_silac_lys8_kdp_per_hour_no_dilution_correction) | vector (1015) | node | 6.9×10⁷ |

#### Viability

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| GeneEssentialitySgd | S. cerevisiae | 1,180 | 5 | 1,329 | is_essential | scalar | node | 6.8×10⁵ |
| SynthLethalityYeastSynthLethDb | S. cerevisiae | 14,000 | 1 | 14,000 | is_synthetic_lethal | scalar | edge | 8.0×10⁶ |
| SynthRescueYeastSynthLethDb | S. cerevisiae | 6,948 | 1 | 6,948 | is_synthetic_rescue | scalar | edge | 4.0×10⁶ |

#### Total

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| **Total (69 datasets, 69 probed)** |  |  |  | **93,266,786** |  |  |  | **7.2×10¹⁰** |

### Signal controls

#### Environmental / chemogenomic response

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| CellYieldDeChiara2022 | 36,286 | 72,572 | 1.7×10⁷ | 1.2×10⁶ | 1.7×10⁷ | 1.7×10⁷ | 1.4×10⁷ | 3.5×10⁴ | 0.07 | 0.99 | 0.00 |
| CrisprMagicLian2019 | 266,304 | 1,331,520 | 1.3×10⁸ | 2.2×10⁷ | 1.3×10⁸ | 1.4×10⁸ | 5.8×10⁷ | 7.4×10⁵ | 0.16 | 0.93 | 0.01 |
| CrispriChemgenSmith2016 | 14,463 | 72,315 | 7.5×10⁶ | 7.3×10⁵ | 7.6×10⁶ | 8.2×10⁶ | 5.1×10⁶ | 2.2×10⁴ | 0.10 | 0.92 | 0.00 |
| CrispriMormino2022 | 12 | 24 | 1.4×10⁴ | 5.9×10³ | 1.4×10⁴ | 1.4×10⁴ | 1.4×10⁴ | 1.8×10² | 0.43 | 0.98 | 0.01 |
| DoublingTimeDeChiara2022 | 36,295 | 72,590 | 1.7×10⁷ | 1.2×10⁶ | 1.7×10⁷ | 1.7×10⁷ | 1.4×10⁷ | 3.6×10⁴ | 0.07 | 0.98 | 0.00 |
| EnvChemgenAuesukaree2009 | 525 | 5,419 | 3.1×10⁵ | 1.3×10⁵ | 3.4×10⁵ | 4.9×10⁵ | 2.9×10⁵ | 1.4×10³ | 0.41 | 0.63 | 0.00 |
| EnvChemgenCostanzo2021 | 61,640 | 1,466,135 | 1.1×10⁸ | 3.3×10⁷ | 1.3×10⁸ | 1.7×10⁸ | 9.9×10⁷ | 2.6×10⁵ | 0.31 | 0.64 | 0.00 |
| EnvChemgenHoepfner2014 | 30,308,574 | 60,617,148 | 2.1×10¹⁰ | 2.4×10⁹ | 2.1×10¹⁰ | 2.1×10¹⁰ | 1.8×10¹⁰ | 3.5×10⁷ | 0.12 | 0.99 | 0.00 |
| EnvChemgenMota2024 | 1,289 | 16,757 | 7.4×10⁵ | 3.6×10⁵ | 8.2×10⁵ | 1.3×10⁶ | 7.1×10⁵ | 2.2×10³ | 0.48 | 0.55 | 0.00 |
| EnvChemgenParsons2006 | 329,254 | 987,762 | 2.2×10⁸ | 2.6×10⁷ | 2.2×10⁸ | 2.4×10⁸ | 1.9×10⁸ | 6.9×10⁵ | 0.12 | 0.94 | 0.00 |
| EnvChemgenPiotrowski2017 | 3,435,820 | 6,871,640 | 1.4×10⁹ | 2.3×10⁸ | 1.4×10⁹ | 1.5×10⁹ | 1.3×10⁹ | 3.9×10⁶ | 0.16 | 0.96 | 0.00 |
| EnvChemgenVanacloig2022 | 164,205 | 656,820 | 9.3×10⁷ | 8.6×10⁶ | 9.3×10⁷ | 9.5×10⁷ | 4.5×10⁷ | 3.6×10⁵ | 0.09 | 0.98 | 0.00 |
| EnvChemgenWildenhain2015 | 449,033 | 898,066 | 2.6×10⁸ | 3.7×10⁷ | 2.6×10⁸ | 2.7×10⁸ | 2.4×10⁸ | 3.6×10⁵ | 0.14 | 0.97 | 0.00 |
| EnvChemgenYoshikawa2009 | 37,832 | 75,664 | 1.0×10⁷ | 3.0×10⁶ | 1.0×10⁷ | 1.1×10⁷ | 9.7×10⁶ | 1.3×10⁴ | 0.30 | 0.90 | 0.00 |
| FattyAcidSmith2006 | 14,310 | 28,620 | 1.8×10⁶ | 6.7×10⁵ | 1.8×10⁶ | 2.3×10⁶ | 1.7×10⁶ | 8.5×10³ | 0.38 | 0.76 | 0.00 |
| HetHillenmeyer2008 | 3,074,132 | 7,586,524 | 2.0×10⁹ | 2.7×10⁸ | 2.0×10⁹ | 2.0×10⁹ | 1.7×10⁹ | 6.3×10⁶ | 0.14 | 0.97 | 0.00 |
| HomHillenmeyer2008 | 1,228,676 | 3,266,922 | 8.4×10⁸ | 1.1×10⁸ | 8.4×10⁸ | 8.6×10⁸ | 7.0×10⁸ | 2.9×10⁶ | 0.13 | 0.97 | 0.00 |
| IsolateGrowthPeter2018 | 33,985 | 67,970 | 1.6×10⁷ | 1.3×10⁶ | 1.6×10⁷ | 1.6×10⁷ | 1.4×10⁷ | 3.5×10⁴ | 0.08 | 0.96 | 0.00 |

#### Expression (RNA-seq)

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Boocock2025 | 79,516 | 360,790,503 | 8.9×10⁹ | 2.4×10⁹ | 9.4×10⁹ | 1.1×10¹⁰ | 3.8×10⁹ | 2.3×10⁸ | 0.27 | 0.82 | 0.03 |
| CaudalPanTranscriptome2024 | 943 | 17,715,967 | 1.7×10⁹ | 3.7×10⁸ | 1.8×10⁹ | 2.1×10⁹ | 1.2×10⁹ | 1.3×10⁷ | 0.21 | 0.84 | 0.01 |
| NadalRibellesPerturbSeq2025 | 6,188 | 36,405,529 | 2.5×10⁹ | 8.2×10⁸ | 2.5×10⁹ | 2.6×10⁹ | 9.4×10⁸ | 3.1×10⁷ | 0.33 | 0.95 | 0.01 |
| Nguessan2025 | 2,951 | 16,459,410 | 4.5×10⁸ | 1.9×10⁸ | 4.7×10⁸ | 5.1×10⁸ | 2.5×10⁸ | 1.1×10⁷ | 0.42 | 0.88 | 0.02 |
| PooledPromoterExpressionVaishnav2022 | 71,158 | 2,638,069 | 1.3×10⁸ | 3.8×10⁷ | 2.0×10⁸ | 2.4×10⁸ | 1.2×10⁸ | 1.4×10⁵ | 0.30 | 0.52 | 0.00 |
| TfDeletionPseudobulkJackson2020 | 784 | 3,029,208 | 9.7×10⁷ | 7.0×10⁷ | 9.8×10⁷ | 9.9×10⁷ | 8.0×10⁷ | 1.9×10⁶ | 0.72 | 0.98 | 0.02 |

#### Expression (microarray)

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| DmMicroarraySameith2015 | 72 | 2,220,912 | 1.7×10⁸ | 4.9×10⁷ | 1.7×10⁸ | 1.7×10⁸ | 5.2×10⁷ | 1.6×10⁶ | 0.29 | 0.98 | 0.01 |
| InductionTimecourseHackett2020 | 1,752 | 84,572,800 | 4.7×10⁹ | 1.8×10⁹ | 4.8×10⁹ | 5.2×10⁹ | 1.9×10⁹ | 5.1×10⁷ | 0.39 | 0.90 | 0.01 |
| MicroarrayKemmeren2014 | 1,484 | 45,775,464 | 3.2×10⁹ | 1.0×10⁹ | 3.3×10⁹ | 3.3×10⁹ | 1.1×10⁹ | 2.7×10⁷ | 0.32 | 0.98 | 0.01 |
| MicroarrayScoreHu2007 | 263 | 3,344,872 | 2.5×10⁸ | 7.5×10⁷ | 2.5×10⁸ | 2.6×10⁸ | 8.1×10⁷ | 3.3×10⁶ | 0.30 | 0.97 | 0.01 |
| SmMicroarraySameith2015 | 82 | 2,529,372 | 1.8×10⁸ | 5.6×10⁷ | 1.8×10⁸ | 1.8×10⁸ | 5.8×10⁷ | 1.5×10⁶ | 0.32 | 0.98 | 0.01 |

#### Fitness

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Bloom2019 | 530,100 | 1,060,200 | 2.6×10⁸ | 2.0×10⁷ | 2.6×10⁸ | 2.7×10⁸ | 2.3×10⁸ | 5.5×10⁵ | 0.08 | 0.97 | 0.00 |
| CrispriSegregantHale2024 | 8,692,635 | 34,770,540 | 6.7×10⁹ | 3.2×10⁸ | 7.3×10⁹ | 7.6×10⁹ | 4.9×10⁹ | 1.5×10⁷ | 0.05 | 0.88 | 0.00 |
| DmfCostanzo2016 | 20,705,612 | 106,311,976 | 7.1×10⁹ | 9.4×10⁸ | 7.3×10⁹ | 8.7×10⁹ | 4.5×10⁹ | 5.5×10⁷ | 0.13 | 0.82 | 0.01 |
| DmfKuzmin2018 | 410,571 | 2,052,827 | 1.5×10⁸ | 2.0×10⁷ | 1.5×10⁸ | 1.8×10⁸ | 1.0×10⁸ | 1.2×10⁶ | 0.14 | 0.85 | 0.01 |
| DmfKuzmin2020 | 632,998 | 3,164,990 | 2.4×10⁸ | 3.2×10⁷ | 2.4×10⁸ | 2.8×10⁸ | 1.6×10⁸ | 1.8×10⁶ | 0.13 | 0.85 | 0.01 |
| SmfBaryshnikova2010 | 6,023 | 24,868 | 1.8×10⁶ | 2.5×10⁵ | 1.8×10⁶ | 2.2×10⁶ | 1.1×10⁶ | 2.0×10⁴ | 0.14 | 0.81 | 0.01 |
| SmfCostanzo2016 | 20,484 | 105,352 | 5.4×10⁶ | 7.9×10⁵ | 6.3×10⁶ | 7.9×10⁶ | 3.9×10⁶ | 6.1×10⁴ | 0.14 | 0.69 | 0.01 |
| SmfKuzmin2018 | 1,539 | 2,895 | 3.9×10⁵ | 5.8×10⁴ | 3.9×10⁵ | 4.3×10⁵ | 3.2×10⁵ | 3.1×10³ | 0.15 | 0.90 | 0.01 |
| SmfKuzmin2020 | 472 | 2,360 | 9.3×10⁴ | 1.5×10⁴ | 1.0×10⁵ | 1.5×10⁵ | 6.7×10⁴ | 2.3×10³ | 0.17 | 0.61 | 0.02 |
| SmfODuibhir2014 | 1,312 | 47,232 | 2.0×10⁶ | 6.2×10⁵ | 3.3×10⁶ | 4.0×10⁶ | 1.8×10⁶ | 3.2×10³ | 0.31 | 0.50 | 0.00 |
| TmfKuzmin2018 | 91,111 | 455,555 | 3.9×10⁷ | 6.6×10⁶ | 4.0×10⁷ | 4.5×10⁷ | 2.7×10⁷ | 2.6×10⁵ | 0.17 | 0.87 | 0.01 |
| TmfKuzmin2020 | 301,798 | 1,508,990 | 1.3×10⁸ | 2.2×10⁷ | 1.3×10⁸ | 1.5×10⁸ | 8.9×10⁷ | 8.1×10⁵ | 0.17 | 0.87 | 0.01 |

#### Gene interaction

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| DmiCostanzo2016 | 20,705,612 | 44,195,140 | 6.0×10⁹ | 1.0×10⁹ | 6.0×10⁹ | 6.5×10⁹ | 4.7×10⁹ | 4.4×10⁷ | 0.17 | 0.92 | 0.01 |
| DmiKuzmin2018 | 410,399 | 820,798 | 1.3×10⁸ | 2.2×10⁷ | 1.3×10⁸ | 1.4×10⁸ | 1.0×10⁸ | 8.1×10⁵ | 0.16 | 0.94 | 0.01 |
| DmiKuzmin2020 | 632,797 | 1,265,594 | 2.1×10⁸ | 3.3×10⁷ | 2.1×10⁸ | 2.2×10⁸ | 1.6×10⁸ | 1.3×10⁶ | 0.16 | 0.94 | 0.01 |
| TmiKuzmin2018 | 91,111 | 182,222 | 3.4×10⁷ | 6.5×10⁶ | 3.4×10⁷ | 3.6×10⁷ | 2.7×10⁷ | 1.8×10⁵ | 0.19 | 0.94 | 0.01 |
| TmiKuzmin2020 | 301,798 | 603,596 | 1.1×10⁸ | 2.2×10⁷ | 1.1×10⁸ | 1.2×10⁸ | 9.0×10⁷ | 6.0×10⁵ | 0.19 | 0.95 | 0.01 |

#### Genome architecture

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| GenomeArchitecturePuddu2019 | 8,835 | 256,215 | 1.2×10⁷ | 5.3×10⁵ | 1.2×10⁷ | 1.6×10⁷ | 3.3×10⁶ | 1.9×10⁵ | 0.05 | 0.72 | 0.02 |

#### Life cycle

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| LifeCycleDeChiara2022 | 4,039 | 16,382 | 1.3×10⁶ | 1.4×10⁵ | 1.4×10⁶ | 1.6×10⁶ | 8.5×10⁵ | 1.5×10⁴ | 0.10 | 0.83 | 0.01 |

#### Metabolite

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| AminoAcidCooper2010 | 4,382 | 136,404 | 5.2×10⁶ | 4.0×10⁵ | 5.2×10⁶ | 6.6×10⁶ | 1.2×10⁶ | 8.8×10⁴ | 0.08 | 0.80 | 0.02 |
| AminoAcidMulleder2016 | 4,678 | 210,510 | 1.1×10⁷ | 1.4×10⁶ | 1.2×10⁷ | 1.5×10⁷ | 3.8×10⁶ | 1.4×10⁵ | 0.12 | 0.74 | 0.01 |
| BetaxanthinCachera2023 | 4,759 | 52,349 | 4.5×10⁶ | 1.9×10⁶ | 5.0×10⁶ | 6.1×10⁶ | 3.5×10⁶ | 2.7×10⁴ | 0.41 | 0.74 | 0.01 |
| FattyAcidXue2025 | 176 | 2,816 | 1.9×10⁵ | 2.8×10⁴ | 1.9×10⁵ | 2.3×10⁵ | 5.9×10⁴ | 2.1×10³ | 0.15 | 0.84 | 0.01 |
| IsobutanolScreenLopez2024 | 4,805 | 14,415 | 9.0×10⁵ | 1.8×10⁵ | 9.2×10⁵ | 1.1×10⁶ | 5.3×10⁵ | 7.4×10³ | 0.20 | 0.81 | 0.01 |
| IsobutanolValidatedLopez2024 | 227 | 908 | 7.4×10⁴ | 1.2×10⁴ | 7.6×10⁴ | 8.6×10⁴ | 3.4×10⁴ | 3.7×10² | 0.16 | 0.86 | 0.00 |
| MetaboliteDaSilveira2014 | 127 | 36,419 | 1.1×10⁶ | 9.9×10⁴ | 1.2×10⁶ | 1.6×10⁶ | 3.2×10⁵ | 1.9×10⁴ | 0.09 | 0.73 | 0.02 |
| MetaboliteZelezniak2018 | 129 | 5,166 | 3.5×10⁵ | 2.7×10⁴ | 3.6×10⁵ | 3.9×10⁵ | 8.3×10⁴ | 4.6×10³ | 0.08 | 0.91 | 0.01 |
| OrganicAcidYoshida2012 | 17 | 391 | 3.3×10⁴ | 9.6×10³ | 3.4×10⁴ | 4.5×10⁴ | 2.7×10⁴ | 5.6×10² | 0.29 | 0.75 | 0.02 |

#### Morphology

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| CarotenoidOzaydin2013 | 4,607 | 179,673 | 7.7×10⁶ | 2.9×10⁶ | 1.3×10⁷ | 1.5×10⁷ | 7.7×10⁶ | 1.1×10⁴ | 0.38 | 0.50 | 0.00 |
| ScmdOhnuki2018 | 1,112 | 564,896 | 4.9×10⁷ | 8.9×10⁵ | 5.0×10⁷ | 5.4×10⁷ | 6.4×10⁶ | 5.4×10⁵ | 0.02 | 0.92 | 0.01 |
| ScmdOhnuki2022 | 1,982 | 994,962 | 8.8×10⁷ | 1.4×10⁶ | 8.9×10⁷ | 9.4×10⁷ | 9.2×10⁶ | 9.3×10⁵ | 0.02 | 0.93 | 0.01 |
| ScmdOhya2005 | 4,718 | 2,363,718 | 1.5×10⁸ | 2.9×10⁶ | 1.5×10⁸ | 1.7×10⁸ | 2.0×10⁷ | 2.3×10⁶ | 0.02 | 0.91 | 0.02 |

#### Protein abundance

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| ProteomeMessner2021 | 70 | 277,682 | 1.2×10⁷ | 1.1×10⁶ | 1.2×10⁷ | 1.3×10⁷ | 4.9×10⁶ | 1.4×10⁵ | 0.09 | 0.97 | 0.01 |
| ProteomeMessner2023 | 4,699 | 16,955,915 | 1.2×10⁹ | 5.2×10⁷ | 1.2×10⁹ | 1.2×10⁹ | 3.5×10⁸ | 8.7×10⁶ | 0.05 | 0.96 | 0.01 |
| ProteomeMuenzner2024 | 795 | 7,685,904 | 9.6×10⁸ | 1.1×10⁸ | 9.8×10⁸ | 1.2×10⁹ | 8.2×10⁸ | 3.4×10⁶ | 0.12 | 0.77 | 0.00 |
| ProteomeZelezniak2018 | 97 | 211,363 | 1.7×10⁷ | 2.8×10⁵ | 1.8×10⁷ | 1.7×10⁷ | 3.3×10⁶ | 1.4×10⁵ | 0.02 | 1.00 | 0.01 |

#### Protein synthesis / turnover

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| ProteinTurnoverMuenzner2024 | 55 | 539,502 | 6.9×10⁷ | 7.2×10⁶ | 7.1×10⁷ | 8.9×10⁷ | 5.7×10⁷ | 2.5×10⁵ | 0.10 | 0.78 | 0.00 |

#### Viability

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| GeneEssentialitySgd | 1,329 | 3 | 6.8×10⁵ | 2.6×10⁵ | 6.8×10⁵ | 6.8×10⁵ | 6.8×10⁵ | 1.7×10² | 0.38 | 1.00 | 0.00 |
| SynthLethalityYeastSynthLethDb | 14,000 | 14,000 | 8.0×10⁶ | 2.3×10⁶ | 8.1×10⁶ | 8.3×10⁶ | 7.9×10⁶ | 1.0×10⁴ | 0.29 | 0.96 | 0.00 |
| SynthRescueYeastSynthLethDb | 6,948 | 0 | 4.0×10⁶ | 1.2×10⁶ | 4.0×10⁶ | 4.0×10⁶ | 4.0×10⁶ | 1.6×10² | 0.29 | 1.00 | 0.00 |

#### Total

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| **Total (69 probed)** | **93,266,786** | **886,671,302** | **7.2×10¹⁰** | **1.3×10¹⁰** | **7.4×10¹⁰** | **7.9×10¹⁰** | **4.8×10¹⁰** | **5.6×10⁸** | **0.18** | **0.91** | **0.01** |

Source JSON: `experiments/database/results/pre-build/2026-10-14/scerevisiae_datasets.json` (probe variant `probes-v1`, gzip level 6, seed 0).
