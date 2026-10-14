---
id: u6wyl2ym63oej64cnums6vu
title: Bacteria Datasets Table
desc: ''
updated: 1791955932313
created: 1791955932313
---

## Bacteria datasets: Instances vs Signal (gzip), with the Signal controls

<!-- GENERATED: raw data from experiments/database/scripts/build_bacteria_datasets_table.py
(-> results/pre-build/<date>/<organisms>_datasets.json); this note is a VIEW rendered by
render_bacteria_datasets_table.py. Do not edit tables by hand. -->

Pre-build inventory of the E. coli + P. putida datasets built as LMDBs in the dev tree
(the set `build_adapter_map(organisms={ecoli, pputida})` serves), a temporary counterpart of
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
| CrisprPineneToleranceNiu2019 | E. coli | 54 | 1 | 54 | environment_response (log2_ratio) | scalar | global | 4.0×10⁴ |
| CrispriChemgenChoe2025 | E. coli | 39,291 | 12 | 470,502 | environment_response (log2_ratio) | scalar | global | 2.4×10⁸ |
| CrispriGeneFitnessWang2018 | E. coli | 13,641 | 5 | 20,158 | environment_response (log2_ratio) | scalar | global | 1.5×10⁷ |
| CrispriGuideFfaEnrichmentFang2025 | E. coli | 15,711 | 1 | 15,711 | environment_response (log2_ratio) | scalar | global | 7.3×10⁶ |
| CrispriGuideFitnessWang2018 | E. coli | 53,203 | 5 | 240,481 | environment_response (log2_ratio) | scalar | global | 1.6×10⁸ |
| CrispriKnockdownCui2018 | E. coli | 71,696 | 1 | 143,392 | environment_response (log2_ratio) | scalar | global | 7.1×10⁷ |
| CrispriScreenRousset2018 | E. coli | 40,592 | 4 | 92,252 | environment_response (log2_ratio) | scalar | global | 4.1×10⁷ |
| DoublingTimeCaglar2017 | E. coli | 1 | 16 | 55 | environment_response (growth_rate) | scalar | global | 3.2×10⁴ |
| EnvChemgenGirgis2009 | E. coli | 3,976 | 17 | 66,287 | environment_response (z_score) | scalar | global | 3.6×10⁷ |
| EnvChemgenNichols2011 | E. coli | 3,967 | 324 | 1,286,962 | environment_response (z_score) | scalar | global | 4.0×10⁸ |
| EnvChemgenShiver2016 | E. coli | 3,846 | 51 | 210,998 | environment_response (z_score) | scalar | global | 9.9×10⁷ |
| EnvChemgenWang2015 | E. coli | 46 | 1 | 46 | environment_response (log2_ratio) | scalar | global | 2.9×10⁴ |
| EnvChemgenWang2024 | E. coli | 4,419 | 6 | 26,514 | environment_response (log2_ratio) | scalar | global | 9.3×10⁶ |
| FitnessBrowserEcoli | E. coli | 3,789 | 3 | 22,734 | environment_response (log2_ratio) | scalar | global | 1.1×10⁷ |
| GrowthRateLamoureux2023 | E. coli | 21 | 54 | 354 | environment_response (growth_rate) | scalar | global | 1.9×10⁵ |
| GrowthRateS23Schmidt2016 | E. coli | 1 | 22 | 26 | environment_response (growth_rate) | scalar | global | 2.0×10⁴ |
| GrowthYieldBaba2006 | E. coli | 3,912 | 3 | 11,735 | environment_response (optical_density) | scalar | global | 3.3×10⁶ |
| MismatchCrispriFitnessHawkins2020 | E. coli | 24,186 | 1 | 24,186 | environment_response (relative_growth_rate) | scalar | global | 1.5×10⁷ |
| PhageRbTnseqMutalik2020 | E. coli | 3,716 | 71 | 287,815 | environment_response (log2_ratio) | scalar | global | 2.1×10⁸ |
| RbTnseqPrice2018Ecoli | E. coli | 3,789 | 104 | 598,662 | environment_response (log2_ratio) | scalar | global | 4.0×10⁸ |
| RbTnseqTeteneva2024 | E. coli | 3,691 | 18 | 66,162 | environment_response (log2_ratio) | scalar | global | 2.3×10⁷ |
| RbTnseqWetmore2015 | E. coli | 3,646 | 85 | 353,662 | environment_response (log2_ratio) | scalar | global | 2.4×10⁸ |
| EnvMetalTnseqRoyet2025 | P. putida | 5,491 | 4 | 21,583 | environment_response (log2_ratio) | scalar | global | 6.7×10⁶ |
| FitnessBrowserPutida | P. putida | 9,556 | 146 | 1,118,052 | environment_response (log2_ratio) | scalar | global | 5.3×10⁸ |
| IsoprenolSelectionMenasalvas2025 | P. putida | 58 | 1 | 58 | environment_response (categorical) | scalar | global | 7.5×10⁴ |
| IsoprenolToleranceLim2025 | P. putida | 3 | 2 | 3 | environment_response (log2_ratio) | scalar | global | 1.8×10⁴ |
| LactamGrowthRateThompson2019 | P. putida | 3 | 3 | 9 | environment_response (growth_rate) | scalar | global | 1.2×10⁴ |
| RbTnseqBorchert2023 | P. putida | 271 | 32 | 10,824 | environment_response (log2_ratio) | scalar | global | 4.5×10⁶ |
| RbTnseqBorchert2024 | P. putida | 9,464 | 155 | 1,466,920 | environment_response (log2_ratio) | scalar | global | 4.2×10⁸ |

#### Expression (RNA-seq)

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| MrnaFractionBalakrishnan2022 | E. coli | 3 | 14 | 28 | mrna_number_fraction (rnaseq_mrna_number_fraction) | vector (4342) | node | 7.2×10⁶ |
| RnaseqCaglar2017 | E. coli | 1 | 57 | 152 | expression_tpm (rnaseq_tpm) | vector (4196) | node | 6.7×10⁷ |
| RnaseqGuzman2018 | E. coli | 7 | 2 | 14 | expression_tpm (rnaseq_tpm) | vector (4329) | node | 4.5×10⁶ |
| RnaseqLamoureux2023 | E. coli | 74 | 119 | 1,035 | expression_tpm (rnaseq_tpm) | vector (4257) | node | 5.8×10⁸ |
| RnaseqPublicK12Lamoureux2023 | E. coli | 72 | 198 | 1,675 | expression_tpm (rnaseq_tpm_from_released_counts) | vector (4355) | node | 9.5×10⁸ |
| PutidaPrecise321Lim2022 | P. putida | 14 | 118 | 321 | expression_tpm (rnaseq_tpm) | vector (5564) | node | 2.4×10⁸ |

#### Fitness

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| CarbonSourceTong2020 | E. coli | 3,714 | 30 | 111,420 | fitness | scalar | global | 4.8×10⁷ |
| CrispriCrossRachwalski2024 | E. coli | 14,209 | 15 | 48,900 | fitness | scalar | global | 1.9×10⁷ |
| GrowthAucRapp2026 | E. coli | 1,515 | 1 | 1,515 | fitness | scalar | global | 7.7×10⁵ |
| GrowthRateCampos2018 | E. coli | 3,664 | 1 | 3,664 | fitness | scalar | global | 1.1×10⁶ |
| GrowthRateChoe2019 | E. coli | 6 | 1 | 8 | fitness | scalar | global | 1.1×10⁴ |
| GrowthRateSchmidt2016 | E. coli | 3 | 2 | 6 | fitness | scalar | global | 7.8×10³ |
| GrowthWang2015 | E. coli | 46 | 1 | 46 | fitness | scalar | global | 2.1×10⁴ |
| TranscriptionFactorKnockoutChoe2019 | E. coli | 2 | 1 | 2 | fitness | scalar | global | 5.3×10³ |
| FitnessContrastBorchert2023 | P. putida | 5,031 | 12 | 64,853 | t_statistic (rbtnseq_gene_fitness_two_sample_t_test) | scalar | global | 2.2×10⁷ |

#### Flux

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| FluxIshii2007 | E. coli | 25 | 5 | 28 | net_flux (c13_mfa_net_flux_percent_of_specific_glucose_uptake) | vector (42) | bipartite node | 6.0×10⁴ |
| FluxLi2021 | E. coli | 3 | 1 | 3 | net_flux (c13_mfa_net_flux_mmol_per_gdw_per_h) | vector (99) | bipartite node | 5.7×10⁴ |

#### Gene interaction

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| GeneInteractionBabu2014 | E. coli | 42,703 | 1 | 42,705 | gene_interaction | scalar | edge | 1.2×10⁷ |
| GeneInteractionButland2008 | E. coli | 314,769 | 1 | 314,769 | gene_interaction | scalar | edge | 8.2×10⁷ |

#### Metabolite

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ExometaboliteBrunk2016 | E. coli | 9 | 28 | 126 | metabolite_level (hplc_extracellular_concentration_g_per_l) | vector (6) | bipartite node | 1.8×10⁵ |
| MetaboliteIntensityRapp2026 | E. coli | 407 | 1 | 407 | metabolite_level (fi_ms_annotated_feature_absolute_intensity_mean_of_2_plates) | vector (2) | bipartite node | 3.9×10⁵ |
| MetabolomeBrunk2016 | E. coli | 9 | 26 | 117 | metabolite_level (lc_ms_concentration_um) | vector (51) | bipartite node | 3.5×10⁵ |
| MetabolomeFuhrer2017 | E. coli | 3,806 | 1 | 3,806 | metabolite_level (fia_tof_ms_ion_modified_z_score) | vector (7534) | bipartite node | 1.9×10⁹ |
| MetabolomeIshii2007 | E. coli | 25 | 5 | 28 | metabolite_level (ce_tofms_intracellular_concentration_mm) | vector (130) | bipartite node | 4.9×10⁵ |
| MetabolomeRapp2026 | E. coli | 1,497 | 1 | 1,497 | metabolite_level (fi_ms_iml1515_feature_fold_change_vs_batch_median_mean_of_2_plates) | vector (1321) | bipartite node | 5.6×10⁸ |
| MetabolomeSchastnaya2021 | E. coli | 103 | 5 | 200 | metabolite_level (fia_tof_ms_ion_log2_fold_change_vs_wild_type) | vector (284) | bipartite node | 3.6×10⁶ |
| TargetedMetabolomeRapp2026 | E. coli | 407 | 1 | 407 | metabolite_level (lc_msms_eic_peak_height_fold_change_vs_library_median) | vector (2) | bipartite node | 3.7×10⁵ |
| MetaboliteGrowthPhaseMenasalvas2025 | P. putida | 3 | 1 | 3 | metabolite_level (lc_ms_intracellular_concentration_uM) | vector (46) | bipartite node | 7.4×10⁴ |
| MetaboliteProductionPhaseMenasalvas2025 | P. putida | 3 | 1 | 3 | metabolite_level (lc_ms_intracellular_concentration_uM) | vector (46) | bipartite node | 7.3×10⁴ |

#### Morphology

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| MorphologyCampos2018 | E. coli | 3,664 | 1 | 3,664 | morphology | vector (15) | global | 8.5×10⁶ |

#### Phosphoproteome

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| PhosphositeIntensityPotel2018 | E. coli | 1 | 2 | 12 | phosphosite_intensity (fe3_imac_lc_msms_maxquant_summed_xic) | vector (989) | node | 1.3×10⁷ |

#### Product titer

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| BiofuelTiterBrunk2016 | E. coli | 8 | 18 | 72 | titer | scalar | global | 1.1×10⁵ |
| IsopentenolTiterFoo2014 | E. coli | 9 | 1 | 9 | titer | scalar | global | 1.5×10⁴ |
| PineneFoldChangeNiu2019 | E. coli | 40 | 1 | 40 | titer | scalar | global | 2.2×10⁴ |
| IsoprenolTiterBanerjee2024 | P. putida | 14 | 9 | 20 | titer | scalar | global | 3.4×10⁴ |
| IsoprenolTiterCarruthers2025 | P. putida | 432 | 2 | 502 | titer | scalar | global | 4.2×10⁵ |
| IsoprenolTiterDeSiqueira2025 | P. putida | 5 | 2 | 10 | titer | scalar | global | 2.1×10⁵ |
| IsoprenolTiterYunus2026 | P. putida | 125 | 1 | 125 | titer | scalar | global | 6.6×10⁴ |
| IsoprenylAcetateTiterKang2026 | P. putida | 11 | 13 | 26 | titer | scalar | global | 6.7×10⁴ |
| ValerolactamTiterThompson2019 | P. putida | 4 | 2 | 8 | titer | scalar | global | 1.7×10⁴ |

#### Promoter activity

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| PromoterReporterMohiuddin2022 | E. coli | 1,809 | 27 | 69,480 | promoter_activity | scalar | node | 2.8×10⁷ |

#### Protein abundance

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ProteinFoldChangeCaglar2017 | E. coli | 1 | 8 | 8 | protein_fold_change (deseq2_wald_log2_fold_change_design_batch_plus_condition) | vector (4180) | node | 1.1×10⁷ |
| ProteomeBrunk2016 | E. coli | 9 | 18 | 81 | protein_abundance (srm_protein_peak_area) | vector (68) | node | 3.6×10⁵ |
| ProteomeCaglar2017 | E. coli | 1 | 40 | 105 | protein_abundance (lcmsms_spectral_count_deseq2_size_factor_normalized) | vector (4196) | node | 2.5×10⁷ |
| ProteomeIshii2007 | E. coli | 25 | 5 | 28 | protein_abundance (lc_ms_ms_absolute_mg_protein_per_g_dry_cell_weight) | vector (57) | node | 2.9×10⁵ |
| ProteomeMori2021 | E. coli | 1 | 1 | 7 | protein_abundance (absolute_protein_mass_fraction_xtop_dia_swath_riboprofiling_scaled) | vector (1897) | node | 1.1×10⁶ |
| ProteomeSchmidt2016 | E. coli | 1 | 21 | 21 | protein_abundance (absolute_protein_copies_per_cell_label_free_sid_anchored) | vector (2359) | node | 1.1×10⁷ |
| ProteomeSrmSet1Schmidt2016 | E. coli | 1 | 18 | 18 | protein_abundance (absolute_protein_copies_per_cell_srm_sid_dataset1_technical_sd) | vector (41) | node | 1.5×10⁵ |
| ProteomeSrmSet2Schmidt2016 | E. coli | 1 | 21 | 21 | protein_abundance (absolute_protein_copies_per_cell_srm_sid_dataset2_biological_sd) | vector (31) | node | 1.3×10⁵ |
| CampaignProteomeCarruthers2025 | P. putida | 401 | 1 | 465 | protein_abundance (dia_nn_top3_percent_of_proteome_mean) | vector (1616) | node | 1.3×10⁸ |
| CrispriArrayYunus2026 | P. putida | 25 | 1 | 25 | protein_fold_change (dia_nn_top3_relative_to_control_strain) | scalar | node | 2.5×10⁴ |
| CrispriDifferentialProteomeYunus2026 | P. putida | 1 | 1 | 1 | protein_fold_change (dia_nn_top3_fold_change_relative_to_control_strain_equal_variance_t_test) | vector (305) | node | 6.1×10⁴ |
| CrispriKnockdownYunus2026 | P. putida | 102 | 1 | 102 | protein_fold_change (dia_nn_top3_relative_to_control_strain) | scalar | node | 4.9×10⁴ |
| CrispriPanelProteomeYunus2026 | P. putida | 125 | 1 | 125 | protein_abundance (dia_nn_top3_signal_benchling_displayed) | vector (207) | node | 1.5×10⁶ |
| ProteomeBanerjee2025 | P. putida | 4 | 3 | 6 | protein_abundance (diann_lfq_log2_intensity_replicate_mean) | vector (2248) | node | 2.8×10⁶ |
| ProteomeCarruthers2025 | P. putida | 17 | 2 | 21 | protein_abundance (dia_nn_top3_peptide_signal_mean) | vector (1424) | node | 5.6×10⁶ |
| ProteomeDeSiqueira2025 | P. putida | 6 | 3 | 17 | protein_abundance (dia_nn_top3_peptide_signal_replicate_mean) | vector (1531) | node | 5.5×10⁶ |
| ProteomeFoldChangeCarruthers2025 | P. putida | 16 | 1 | 16 | protein_fold_change (dia_log2_fold_change_paired_two_tailed_t_test) | vector (350) | node | 1.5×10⁶ |
| ProteomeFoldChangeLim2025 | P. putida | 2 | 2 | 4 | protein_fold_change (dia_nn_top3_log2_fold_change_welch_two_sample_t_test) | vector (2361) | node | 3.7×10⁶ |
| ProteomeLim2025 | P. putida | 3 | 1 | 3 | protein_abundance (dia_nn_top3_log2_mean) | vector (2361) | node | 1.5×10⁶ |
| ProteomeLog10PercentDeSiqueira2025 | P. putida | 6 | 3 | 17 | protein_abundance (dia_nn_top3_log10_percent_of_total_abundance_replicate_mean) | vector (1531) | node | 6.0×10⁶ |
| ProteomeMenasalvas2025 | P. putida | 3 | 2 | 6 | protein_abundance (dia_protein_counts_sum_mean) | vector (2090) | node | 2.6×10⁶ |
| ProteomePercentDeSiqueira2025 | P. putida | 6 | 3 | 17 | protein_abundance (dia_nn_top3_percent_of_total_abundance_replicate_mean) | vector (1531) | node | 6.0×10⁶ |

#### Protein synthesis / turnover

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ProteinSynthesisRateLi2014 | E. coli | 1 | 3 | 3 | synthesis_rate (ribosome_profiling_footprint_density_absolute_synthesis_rate) | vector (3041) | node | 3.7×10⁵ |
| ProteinTurnoverGupta2024 | E. coli | 6 | 8 | 13 | degradation_rate (n15_ammonium_tmtproc_total_turnover_rate_per_hour) | vector (2553) | node | 1.1×10⁷ |

#### Viability

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| GeneEssentialityBaba2006 | E. coli | 4,288 | 1 | 4,288 | is_essential | scalar | node | 1.8×10⁶ |
| GeneEssentialityGerdes2003 | E. coli | 3,739 | 1 | 3,739 | is_essential | scalar | node | 1.9×10⁶ |
| GeneEssentialityGoodall2018 | E. coli | 8,309 | 2 | 8,309 | is_essential | scalar | node | 5.1×10⁶ |
| GeneEssentialityPrice2018Ecoli | E. coli | 324 | 1 | 324 | is_essential | scalar | node | 2.6×10⁵ |
| GradedGeneEssentialityGoodall2018 | E. coli | 317 | 2 | 317 | essentiality_call | scalar | node | 2.2×10⁵ |
| GradedGeneEssentialityRoyet2025 | P. putida | 5,729 | 2 | 11,458 | essentiality_call | scalar | node | 4.8×10⁶ |

#### Total

| Dataset | Host | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| **Total (100 datasets, 100 probed)** |  |  |  | **7,257,468** |  |  |  | **7.7×10⁹** |

### Signal controls

#### Environmental / chemogenomic response

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| CrisprPineneToleranceNiu2019 | 54 | 108 | 4.0×10⁴ | 7.8×10³ | 4.0×10⁴ | 4.2×10⁴ | 3.5×10⁴ | 2.7×10² | 0.20 | 0.94 | 0.01 |
| CrispriChemgenChoe2025 | 470,502 | 1,411,506 | 2.4×10⁸ | 2.9×10⁷ | 2.5×10⁸ | 2.5×10⁸ | 1.9×10⁸ | 8.3×10⁵ | 0.12 | 0.94 | 0.00 |
| CrispriGeneFitnessWang2018 | 20,158 | 121,191 | 1.5×10⁷ | 2.2×10⁶ | 1.5×10⁷ | 1.7×10⁷ | 8.7×10⁶ | 1.1×10⁵ | 0.14 | 0.92 | 0.01 |
| CrispriGuideFfaEnrichmentFang2025 | 15,711 | 47,613 | 7.3×10⁶ | 1.7×10⁶ | 7.3×10⁶ | 7.8×10⁶ | 5.9×10⁶ | 2.4×10⁴ | 0.23 | 0.93 | 0.00 |
| CrispriGuideFitnessWang2018 | 240,481 | 963,737 | 1.6×10⁸ | 2.6×10⁷ | 1.6×10⁸ | 1.7×10⁸ | 1.2×10⁸ | 4.2×10⁵ | 0.16 | 0.94 | 0.00 |
| CrispriKnockdownCui2018 | 143,392 | 441,964 | 7.1×10⁷ | 1.0×10⁷ | 7.1×10⁷ | 7.7×10⁷ | 5.7×10⁷ | 1.9×10⁵ | 0.15 | 0.93 | 0.00 |
| CrispriScreenRousset2018 | 92,252 | 300,128 | 4.1×10⁷ | 5.5×10⁶ | 4.2×10⁷ | 4.5×10⁷ | 3.0×10⁷ | 9.3×10⁴ | 0.13 | 0.91 | 0.00 |
| DoublingTimeCaglar2017 | 55 | 275 | 3.2×10⁴ | 5.5×10³ | 3.2×10⁴ | 3.5×10⁴ | 2.0×10⁴ | 4.5×10² | 0.17 | 0.90 | 0.01 |
| EnvChemgenGirgis2009 | 66,287 | 132,574 | 3.6×10⁷ | 4.8×10⁶ | 3.7×10⁷ | 3.9×10⁷ | 3.4×10⁷ | 9.1×10⁴ | 0.13 | 0.94 | 0.00 |
| EnvChemgenNichols2011 | 1,286,962 | 2,573,924 | 4.0×10⁸ | 7.2×10⁷ | 4.0×10⁸ | 4.2×10⁸ | 3.4×10⁸ | 1.5×10⁶ | 0.18 | 0.96 | 0.00 |
| EnvChemgenShiver2016 | 210,998 | 210,998 | 9.9×10⁷ | 1.7×10⁷ | 9.9×10⁷ | 10.0×10⁷ | 8.2×10⁷ | 2.1×10⁵ | 0.18 | 0.99 | 0.00 |
| EnvChemgenWang2015 | 46 | 92 | 2.9×10⁴ | 7.4×10³ | 2.9×10⁴ | 2.9×10⁴ | 2.4×10⁴ | 2.6×10² | 0.26 | 0.98 | 0.01 |
| EnvChemgenWang2024 | 26,514 | 106,056 | 9.3×10⁶ | 1.9×10⁶ | 9.4×10⁶ | 1.1×10⁷ | 7.3×10⁶ | 6.7×10⁴ | 0.20 | 0.85 | 0.01 |
| FitnessBrowserEcoli | 22,734 | 68,202 | 1.1×10⁷ | 1.5×10⁶ | 1.1×10⁷ | 1.1×10⁷ | 4.6×10⁶ | 6.0×10⁴ | 0.13 | 0.98 | 0.01 |
| GrowthRateLamoureux2023 | 354 | 354 | 1.9×10⁵ | 3.8×10⁴ | 1.9×10⁵ | 2.0×10⁵ | 1.8×10⁵ | 5.3×10² | 0.20 | 0.98 | 0.00 |
| GrowthRateS23Schmidt2016 | 26 | 92 | 2.0×10⁴ | 4.6×10³ | 2.0×10⁴ | 2.3×10⁴ | 1.8×10⁴ | 2.5×10² | 0.23 | 0.89 | 0.01 |
| GrowthYieldBaba2006 | 11,735 | 11,735 | 3.3×10⁶ | 1.4×10⁶ | 3.3×10⁶ | 3.4×10⁶ | 3.0×10⁶ | 8.7×10³ | 0.41 | 0.96 | 0.00 |
| MismatchCrispriFitnessHawkins2020 | 24,186 | 117,184 | 1.5×10⁷ | 1.5×10⁶ | 1.5×10⁷ | 1.6×10⁷ | 8.1×10⁶ | 6.3×10⁴ | 0.10 | 0.92 | 0.00 |
| PhageRbTnseqMutalik2020 | 287,815 | 1,151,260 | 2.1×10⁸ | 2.4×10⁷ | 2.3×10⁸ | 2.4×10⁸ | 1.2×10⁸ | 7.1×10⁵ | 0.11 | 0.90 | 0.00 |
| RbTnseqPrice2018Ecoli | 598,662 | 2,394,648 | 4.0×10⁸ | 5.0×10⁷ | 4.5×10⁸ | 4.7×10⁸ | 2.2×10⁸ | 1.5×10⁶ | 0.12 | 0.86 | 0.00 |
| RbTnseqTeteneva2024 | 66,162 | 132,324 | 2.3×10⁷ | 3.7×10⁶ | 2.3×10⁷ | 2.4×10⁷ | 1.7×10⁷ | 8.9×10⁴ | 0.16 | 0.96 | 0.00 |
| RbTnseqWetmore2015 | 353,662 | 1,414,648 | 2.4×10⁸ | 2.9×10⁷ | 2.6×10⁸ | 2.7×10⁸ | 1.3×10⁸ | 8.4×10⁵ | 0.12 | 0.90 | 0.00 |
| EnvMetalTnseqRoyet2025 | 21,583 | 86,332 | 6.7×10⁶ | 1.5×10⁶ | 6.7×10⁶ | 8.2×10⁶ | 5.5×10⁶ | 4.9×10⁴ | 0.23 | 0.82 | 0.01 |
| FitnessBrowserPutida | 1,118,052 | 3,354,156 | 5.3×10⁸ | 7.2×10⁷ | 5.3×10⁸ | 5.4×10⁸ | 2.1×10⁸ | 3.1×10⁶ | 0.13 | 0.98 | 0.01 |
| IsoprenolSelectionMenasalvas2025 | 58 | 466 | 7.5×10⁴ | 2.8×10⁴ | 7.5×10⁴ | 9.0×10⁴ | 7.2×10⁴ | 1.9×10² | 0.37 | 0.83 | 0.00 |
| IsoprenolToleranceLim2025 | 3 | 26 | 1.8×10⁴ | 6.4×10³ | 1.8×10⁴ | 1.9×10⁴ | 1.8×10⁴ | 1.9×10² | 0.35 | 0.96 | 0.01 |
| LactamGrowthRateThompson2019 | 9 | 9 | 1.2×10⁴ | 4.6×10³ | 1.2×10⁴ | 1.2×10⁴ | 1.1×10⁴ | 1.8×10² | 0.39 | 0.99 | 0.02 |
| RbTnseqBorchert2023 | 10,824 | 10,824 | 4.5×10⁶ | 1.1×10⁶ | 4.5×10⁶ | 4.5×10⁶ | 3.4×10⁶ | 9.2×10³ | 0.25 | 0.99 | 0.00 |
| RbTnseqBorchert2024 | 1,466,920 | 2,420,011 | 4.2×10⁸ | 1.1×10⁸ | 4.2×10⁸ | 4.3×10⁸ | 2.8×10⁸ | 2.4×10⁶ | 0.26 | 0.98 | 0.01 |

#### Expression (RNA-seq)

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| MrnaFractionBalakrishnan2022 | 28 | 121,604 | 7.2×10⁶ | 2.4×10⁶ | 7.2×10⁶ | 8.3×10⁶ | 2.7×10⁶ | 1.1×10⁵ | 0.33 | 0.86 | 0.01 |
| RnaseqCaglar2017 | 152 | 1,275,584 | 6.7×10⁷ | 2.5×10⁷ | 7.1×10⁷ | 7.2×10⁷ | 2.7×10⁷ | 9.5×10⁵ | 0.37 | 0.93 | 0.01 |
| RnaseqGuzman2018 | 14 | 60,678 | 4.5×10⁶ | 1.3×10⁶ | 4.5×10⁶ | 4.5×10⁶ | 1.5×10⁶ | 6.0×10⁴ | 0.29 | 0.98 | 0.01 |
| RnaseqLamoureux2023 | 1,035 | 8,811,990 | 5.8×10⁸ | 1.7×10⁸ | 5.9×10⁸ | 5.8×10⁸ | 1.9×10⁸ | 7.3×10⁶ | 0.29 | 1.00 | 0.01 |
| RnaseqPublicK12Lamoureux2023 | 1,675 | 14,590,925 | 9.5×10⁸ | 2.8×10⁸ | 9.7×10⁸ | 9.5×10⁸ | 3.0×10⁸ | 1.1×10⁷ | 0.30 | 0.99 | 0.01 |
| PutidaPrecise321Lim2022 | 321 | 3,572,400 | 2.4×10⁸ | 7.0×10⁷ | 2.4×10⁸ | 2.4×10⁸ | 7.7×10⁷ | 2.2×10⁶ | 0.29 | 1.00 | 0.01 |

#### Fitness

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| CarbonSourceTong2020 | 111,420 | 220,740 | 4.8×10⁷ | 3.6×10⁶ | 4.8×10⁷ | 5.0×10⁷ | 4.0×10⁷ | 1.1×10⁵ | 0.07 | 0.96 | 0.00 |
| CrispriCrossRachwalski2024 | 48,900 | 136,563 | 1.9×10⁷ | 4.0×10⁶ | 1.9×10⁷ | 2.1×10⁷ | 1.4×10⁷ | 6.6×10⁴ | 0.21 | 0.91 | 0.00 |
| GrowthAucRapp2026 | 1,515 | 7,575 | 7.7×10⁵ | 4.9×10⁴ | 7.7×10⁵ | 8.3×10⁵ | 3.5×10⁵ | 5.1×10³ | 0.06 | 0.92 | 0.01 |
| GrowthRateCampos2018 | 3,664 | 7,328 | 1.1×10⁶ | 1.5×10⁵ | 1.1×10⁶ | 1.1×10⁶ | 7.4×10⁵ | 4.2×10³ | 0.14 | 0.97 | 0.00 |
| GrowthRateChoe2019 | 8 | 32 | 1.1×10⁴ | 2.9×10³ | 1.1×10⁴ | 1.1×10⁴ | 8.1×10³ | 1.9×10² | 0.28 | 0.99 | 0.02 |
| GrowthRateSchmidt2016 | 6 | 24 | 7.8×10³ | 2.5×10³ | 7.8×10³ | 7.9×10³ | 6.0×10³ | 1.8×10² | 0.32 | 0.99 | 0.02 |
| GrowthWang2015 | 46 | 184 | 2.1×10⁴ | 3.7×10³ | 2.0×10⁴ | 2.7×10⁴ | 1.2×10⁴ | 3.5×10² | 0.18 | 0.76 | 0.02 |
| TranscriptionFactorKnockoutChoe2019 | 2 | 2 | 5.3×10³ | 2.4×10³ | 5.3×10³ | 5.3×10³ | 5.2×10³ | 1.7×10² | 0.45 | 1.00 | 0.03 |
| FitnessContrastBorchert2023 | 64,853 | 389,118 | 2.2×10⁷ | 2.1×10⁶ | 2.8×10⁷ | 3.3×10⁷ | 9.2×10⁶ | 1.4×10⁵ | 0.10 | 0.66 | 0.01 |

#### Flux

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| FluxIshii2007 | 28 | 1,200 | 6.0×10⁴ | 10.0×10³ | 6.2×10⁴ | 7.0×10⁴ | 2.7×10⁴ | 1.4×10³ | 0.17 | 0.86 | 0.02 |
| FluxLi2021 | 3 | 1,104 | 5.7×10⁴ | 8.5×10³ | 6.1×10⁴ | 9.8×10⁴ | 1.8×10⁴ | 1.3×10³ | 0.15 | 0.58 | 0.02 |

#### Gene interaction

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| GeneInteractionBabu2014 | 42,705 | 85,410 | 1.2×10⁷ | 2.8×10⁶ | 1.2×10⁷ | 1.2×10⁷ | 9.7×10⁶ | 4.4×10⁴ | 0.25 | 0.93 | 0.00 |
| GeneInteractionButland2008 | 314,769 | 629,538 | 8.2×10⁷ | 2.3×10⁷ | 8.2×10⁷ | 8.7×10⁷ | 6.6×10⁷ | 4.6×10⁵ | 0.28 | 0.94 | 0.01 |

#### Metabolite

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| ExometaboliteBrunk2016 | 126 | 2,450 | 1.8×10⁵ | 3.7×10⁴ | 1.8×10⁵ | 2.2×10⁵ | 1.4×10⁵ | 1.9×10³ | 0.21 | 0.78 | 0.01 |
| MetaboliteIntensityRapp2026 | 407 | 4,532 | 3.9×10⁵ | 1.3×10⁵ | 3.9×10⁵ | 4.5×10⁵ | 2.7×10⁵ | 4.1×10³ | 0.32 | 0.88 | 0.01 |
| MetabolomeBrunk2016 | 117 | 8,511 | 3.5×10⁵ | 5.2×10⁴ | 3.5×10⁵ | 4.8×10⁵ | 1.8×10⁵ | 5.0×10³ | 0.15 | 0.71 | 0.01 |
| MetabolomeFuhrer2017 | 3,806 | 57,348,808 | 1.9×10⁹ | 1.1×10⁹ | 1.9×10⁹ | 2.1×10⁹ | 1.1×10⁹ | 2.9×10⁷ | 0.56 | 0.90 | 0.02 |
| MetabolomeIshii2007 | 28 | 8,080 | 4.9×10⁵ | 5.1×10⁴ | 4.9×10⁵ | 5.8×10⁵ | 1.1×10⁵ | 4.7×10³ | 0.10 | 0.86 | 0.01 |
| MetabolomeRapp2026 | 1,497 | 5,934,108 | 5.6×10⁸ | 1.1×10⁸ | 5.6×10⁸ | 5.8×10⁸ | 1.9×10⁸ | 3.7×10⁶ | 0.19 | 0.96 | 0.01 |
| MetabolomeSchastnaya2021 | 200 | 116,436 | 3.6×10⁶ | 4.7×10⁵ | 3.7×10⁶ | 5.3×10⁶ | 1.6×10⁶ | 6.2×10⁴ | 0.13 | 0.68 | 0.02 |
| TargetedMetabolomeRapp2026 | 407 | 2,899 | 3.7×10⁵ | 1.0×10⁵ | 3.7×10⁵ | 3.8×10⁵ | 2.4×10⁵ | 2.4×10³ | 0.28 | 0.96 | 0.01 |
| MetaboliteGrowthPhaseMenasalvas2025 | 3 | 448 | 7.4×10⁴ | 1.4×10⁴ | 7.5×10⁴ | 8.0×10⁴ | 6.4×10⁴ | 4.8×10² | 0.20 | 0.92 | 0.01 |
| MetaboliteProductionPhaseMenasalvas2025 | 3 | 450 | 7.3×10⁴ | 1.5×10⁴ | 7.4×10⁴ | 8.0×10⁴ | 6.4×10⁴ | 4.9×10² | 0.20 | 0.92 | 0.01 |

#### Morphology

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| MorphologyCampos2018 | 3,664 | 97,234 | 8.5×10⁶ | 1.8×10⁵ | 8.5×10⁶ | 9.5×10⁶ | 1.1×10⁶ | 9.2×10⁴ | 0.02 | 0.90 | 0.01 |

#### Phosphoproteome

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| PhosphositeIntensityPotel2018 | 12 | 141,757 | 1.3×10⁷ | 2.7×10⁶ | 1.3×10⁷ | 1.4×10⁷ | 7.3×10⁶ | 1.5×10⁴ | 0.21 | 0.94 | 0.00 |

#### Product titer

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| BiofuelTiterBrunk2016 | 72 | 702 | 1.1×10⁵ | 3.0×10⁴ | 1.1×10⁵ | 1.3×10⁵ | 1.0×10⁵ | 4.0×10² | 0.28 | 0.84 | 0.00 |
| IsopentenolTiterFoo2014 | 9 | 36 | 1.5×10⁴ | 5.4×10³ | 1.5×10⁴ | 1.6×10⁴ | 1.5×10⁴ | 2.0×10² | 0.35 | 0.94 | 0.01 |
| PineneFoldChangeNiu2019 | 40 | 160 | 2.2×10⁴ | 6.1×10³ | 2.2×10⁴ | 2.9×10⁴ | 2.1×10⁴ | 3.2×10² | 0.28 | 0.77 | 0.01 |
| IsoprenolTiterBanerjee2024 | 20 | 159 | 3.4×10⁴ | 1.1×10⁴ | 3.4×10⁴ | 3.8×10⁴ | 3.3×10⁴ | 3.0×10² | 0.32 | 0.89 | 0.01 |
| IsoprenolTiterCarruthers2025 | 502 | 6,105 | 4.2×10⁵ | 1.2×10⁵ | 4.2×10⁵ | 5.4×10⁵ | 2.7×10⁵ | 2.9×10³ | 0.30 | 0.77 | 0.01 |
| IsoprenolTiterDeSiqueira2025 | 10 | 1,182 | 2.1×10⁵ | 3.1×10⁴ | 2.1×10⁵ | 2.3×10⁵ | 1.8×10⁵ | 1.4×10³ | 0.15 | 0.90 | 0.01 |
| IsoprenolTiterYunus2026 | 125 | 500 | 6.6×10⁴ | 1.4×10⁴ | 6.7×10⁴ | 7.6×10⁴ | 5.9×10⁴ | 2.9×10² | 0.22 | 0.87 | 0.00 |
| IsoprenylAcetateTiterKang2026 | 26 | 329 | 6.7×10⁴ | 1.9×10⁴ | 6.7×10⁴ | 7.5×10⁴ | 6.5×10⁴ | 3.1×10² | 0.29 | 0.89 | 0.00 |
| ValerolactamTiterThompson2019 | 8 | 40 | 1.7×10⁴ | 6.6×10³ | 1.7×10⁴ | 1.8×10⁴ | 1.7×10⁴ | 2.0×10² | 0.38 | 0.96 | 0.01 |

#### Promoter activity

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| PromoterReporterMohiuddin2022 | 69,480 | 208,440 | 2.8×10⁷ | 3.6×10⁶ | 2.8×10⁷ | 3.2×10⁷ | 2.6×10⁷ | 4.1×10⁴ | 0.13 | 0.88 | 0.00 |

#### Protein abundance

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| ProteinFoldChangeCaglar2017 | 8 | 154,960 | 1.1×10⁷ | 3.0×10⁶ | 1.2×10⁷ | 1.2×10⁷ | 3.2×10⁶ | 1.1×10⁵ | 0.27 | 0.91 | 0.01 |
| ProteomeBrunk2016 | 81 | 11,619 | 3.6×10⁵ | 3.4×10⁴ | 3.6×10⁵ | 5.3×10⁵ | 1.3×10⁵ | 6.8×10³ | 0.09 | 0.68 | 0.02 |
| ProteomeCaglar2017 | 105 | 881,160 | 2.5×10⁷ | 1.7×10⁷ | 3.7×10⁷ | 4.2×10⁷ | 1.8×10⁷ | 3.8×10⁵ | 0.67 | 0.60 | 0.01 |
| ProteomeIshii2007 | 28 | 4,722 | 2.9×10⁵ | 1.2×10⁴ | 3.0×10⁵ | 3.5×10⁵ | 3.9×10⁴ | 3.9×10³ | 0.04 | 0.81 | 0.01 |
| ProteomeMori2021 | 7 | 26,716 | 1.1×10⁶ | 1.1×10⁵ | 1.1×10⁶ | 1.3×10⁶ | 4.0×10⁵ | 1.4×10⁴ | 0.10 | 0.84 | 0.01 |
| ProteomeSchmidt2016 | 21 | 145,005 | 1.1×10⁷ | 2.7×10⁵ | 1.1×10⁷ | 1.1×10⁷ | 1.5×10⁶ | 9.0×10⁴ | 0.02 | 1.00 | 0.01 |
| ProteomeSrmSet1Schmidt2016 | 18 | 2,214 | 1.5×10⁵ | 4.7×10³ | 1.5×10⁵ | 1.6×10⁵ | 2.1×10⁴ | 2.0×10³ | 0.03 | 0.92 | 0.01 |
| ProteomeSrmSet2Schmidt2016 | 21 | 1,953 | 1.3×10⁵ | 4.1×10³ | 1.3×10⁵ | 1.4×10⁵ | 2.0×10⁴ | 1.9×10³ | 0.03 | 0.93 | 0.01 |
| CampaignProteomeCarruthers2025 | 465 | 1,978,591 | 1.3×10⁸ | 7.8×10⁶ | 1.3×10⁸ | 1.5×10⁸ | 2.9×10⁷ | 1.4×10⁶ | 0.06 | 0.88 | 0.01 |
| CrispriArrayYunus2026 | 25 | 225 | 2.5×10⁴ | 7.5×10³ | 2.5×10⁴ | 2.8×10⁴ | 1.8×10⁴ | 3.9×10² | 0.30 | 0.88 | 0.02 |
| CrispriDifferentialProteomeYunus2026 | 1 | 916 | 6.1×10⁴ | 1.1×10⁴ | 6.1×10⁴ | 6.6×10⁴ | 1.8×10⁴ | 1.9×10² | 0.19 | 0.92 | 0.00 |
| CrispriKnockdownYunus2026 | 102 | 306 | 4.9×10⁴ | 1.5×10⁴ | 4.9×10⁴ | 5.2×10⁴ | 4.5×10⁴ | 2.4×10² | 0.31 | 0.93 | 0.00 |
| CrispriPanelProteomeYunus2026 | 125 | 51,875 | 1.5×10⁶ | 7.0×10⁴ | 1.5×10⁶ | 2.1×10⁶ | 2.9×10⁵ | 2.8×10⁴ | 0.05 | 0.69 | 0.02 |
| ProteomeBanerjee2025 | 6 | 40,239 | 2.8×10⁶ | 1.5×10⁵ | 2.8×10⁶ | 3.0×10⁶ | 6.8×10⁵ | 2.8×10⁴ | 0.05 | 0.91 | 0.01 |
| ProteomeCarruthers2025 | 21 | 81,310 | 5.6×10⁶ | 2.6×10⁵ | 5.7×10⁶ | 6.3×10⁶ | 1.3×10⁶ | 5.2×10⁴ | 0.05 | 0.90 | 0.01 |
| ProteomeDeSiqueira2025 | 17 | 79,637 | 5.5×10⁶ | 6.6×10⁵ | 5.6×10⁶ | 6.2×10⁶ | 1.5×10⁶ | 5.4×10⁴ | 0.12 | 0.89 | 0.01 |
| ProteomeFoldChangeCarruthers2025 | 16 | 17,970 | 1.5×10⁶ | 1.3×10⁵ | 1.5×10⁶ | 1.5×10⁶ | 2.6×10⁵ | 1.3×10⁴ | 0.09 | 0.98 | 0.01 |
| ProteomeFoldChangeLim2025 | 4 | 47,251 | 3.7×10⁶ | 10.0×10⁵ | 3.8×10⁶ | 4.0×10⁶ | 1.1×10⁶ | 3.8×10⁴ | 0.27 | 0.93 | 0.01 |
| ProteomeLim2025 | 3 | 21,240 | 1.5×10⁶ | 4.5×10⁵ | 1.5×10⁶ | 1.6×10⁶ | 4.9×10⁵ | 1.5×10⁴ | 0.30 | 0.93 | 0.01 |
| ProteomeLog10PercentDeSiqueira2025 | 17 | 79,637 | 6.0×10⁶ | 6.6×10⁵ | 6.0×10⁶ | 6.2×10⁶ | 1.6×10⁶ | 5.4×10⁴ | 0.11 | 0.96 | 0.01 |
| ProteomeMenasalvas2025 | 6 | 37,964 | 2.6×10⁶ | 3.6×10⁵ | 2.7×10⁶ | 3.0×10⁶ | 7.5×10⁵ | 2.6×10⁴ | 0.14 | 0.89 | 0.01 |
| ProteomePercentDeSiqueira2025 | 17 | 79,637 | 6.0×10⁶ | 6.6×10⁵ | 6.0×10⁶ | 6.2×10⁶ | 1.5×10⁶ | 5.4×10⁴ | 0.11 | 0.96 | 0.01 |

#### Protein synthesis / turnover

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| ProteinSynthesisRateLi2014 | 3 | 8,647 | 3.7×10⁵ | 1.7×10⁵ | 3.7×10⁵ | 4.2×10⁵ | 2.0×10⁵ | 7.7×10³ | 0.45 | 0.89 | 0.02 |
| ProteinTurnoverGupta2024 | 13 | 141,301 | 1.1×10⁷ | 2.9×10⁶ | 1.1×10⁷ | 1.1×10⁷ | 3.6×10⁶ | 9.6×10⁴ | 0.27 | 0.99 | 0.01 |

#### Viability

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| GeneEssentialityBaba2006 | 4,288 | 17,152 | 1.8×10⁶ | 5.0×10⁵ | 1.8×10⁶ | 2.1×10⁶ | 1.8×10⁶ | 9.0×10³ | 0.28 | 0.86 | 0.00 |
| GeneEssentialityGerdes2003 | 3,739 | 13,858 | 1.9×10⁶ | 5.4×10⁵ | 1.9×10⁶ | 1.9×10⁶ | 1.6×10⁶ | 1.2×10⁴ | 0.28 | 1.00 | 0.01 |
| GeneEssentialityGoodall2018 | 8,309 | 24,927 | 5.1×10⁶ | 1.1×10⁶ | 5.1×10⁶ | 5.5×10⁶ | 3.9×10⁶ | 1.8×10⁴ | 0.22 | 0.93 | 0.00 |
| GeneEssentialityPrice2018Ecoli | 324 | 2,916 | 2.6×10⁵ | 5.2×10⁴ | 2.6×10⁵ | 3.3×10⁵ | 2.2×10⁵ | 2.6×10³ | 0.20 | 0.80 | 0.01 |
| GradedGeneEssentialityGoodall2018 | 317 | 951 | 2.2×10⁵ | 4.6×10⁴ | 2.2×10⁵ | 2.3×10⁵ | 1.7×10⁵ | 1.1×10³ | 0.22 | 0.93 | 0.01 |
| GradedGeneEssentialityRoyet2025 | 11,458 | 103,122 | 4.8×10⁶ | 9.4×10⁵ | 4.9×10⁶ | 6.5×10⁶ | 3.6×10⁶ | 6.7×10⁴ | 0.19 | 0.74 | 0.01 |

#### Total

| Dataset | Instances | Numeric leaves | Signal (gzip, bits) | Structure (bits) | Shuffled (bits) | Noise (bits) | Binary (bits) | Packed 1-bit (bits) | Structure/Signal | Signal/Noise | Packed/Signal |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| **Total (100 probed)** | **7,257,468** | **115,325,823** | **7.7×10⁹** | **2.3×10⁹** | **7.9×10⁹** | **8.2×10⁹** | **4.1×10⁹** | **7.0×10⁷** | **0.29** | **0.94** | **0.01** |

Source JSON: `experiments/database/results/pre-build/2026-10-14/ecoli-pputida_datasets.json` (probe variant `probes-v1`, gzip level 6, seed 0).
