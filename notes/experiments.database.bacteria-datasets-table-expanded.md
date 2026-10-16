---
id: z7pvc76cin7vqi1f6blhsrq
title: Bacteria Datasets Table Expanded
desc: ''
updated: 1792110891672
created: 1792110891672
---

## Bacteria datasets, expanded: the main table with a description of each dataset

<!-- GENERATED: raw data from experiments/database/scripts/build_bacteria_datasets_table.py
(-> results/pre-build/<date>/<organisms>_datasets.json); this note is a VIEW rendered by
render_bacteria_datasets_table.py. Do not edit tables by hand. -->

The table of [[experiments.database.bacteria-datasets-table]] with one more column.
*Description* is the summary paragraph of the dataset class's docstring in
`torchcell/datasets/`, attached by the build script when it writes the JSON, so a better
description is made by improving that docstring and rerunning both scripts. Every other
column is read from the built LMDB; definitions and the Signal controls are in the
non-expanded note. Commentary: [[experiments.database.scripts.build_bacteria_datasets_table]].

### Datasets

#### Environmental / chemogenomic response

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| CrisprPineneToleranceNiu2019 | E. coli | Niu 2019 CRISPRa/CRISPRi growth ratios under 0.5% pinene in BW25113(PT5-dxs). | 54 | 1 | 54 | environment_response (log2_ratio) | scalar | global | 4.0×10⁴ |
| CrispriChemgenChoe2025 | E. coli | Choe 2025 per-guide CRISPRi abundance response to twelve antibiotics. | 39,291 | 12 | 470,502 | environment_response (log2_ratio) | scalar | global | 2.4×10⁸ |
| CrispriGeneFitnessWang2018 | E. coli | Wang 2018 gene-level CRISPRi calls (Supplementary Data 5) across five screens. | 13,641 | 5 | 20,158 | environment_response (log2_ratio) | scalar | global | 1.5×10⁷ |
| CrispriGuideFfaEnrichmentFang2025 | E. coli | Fang 2025 per-guide CRISPRi-FACS enrichment across the paper's two rounds. | 15,711 | 1 | 15,711 | environment_response (log2_ratio) | scalar | global | 7.3×10⁶ |
| CrispriGuideFitnessWang2018 | E. coli | Wang 2018 per-guide CRISPRi fitness across the paper's five pooled screens. | 53,203 | 5 | 240,481 | environment_response (log2_ratio) | scalar | global | 1.6×10⁸ |
| CrispriKnockdownCui2018 | E. coli | Cui 2018 per-guide dCas9 knockdown log2 fold changes in two MG1655 strains. | 71,696 | 1 | 143,392 | environment_response (log2_ratio) | scalar | global | 7.1×10⁷ |
| CrispriScreenRousset2018 | E. coli | Rousset 2018 per-sgRNA log2FC: growth, three phage challenges, transduction. | 40,592 | 4 | 92,252 | environment_response (log2_ratio) | scalar | global | 4.1×10⁷ |
| DoublingTimeCaglar2017 | E. coli | Caglar 2017 Table S5: 55 absolute doubling times, one per replicate curve. | 1 | 16 | 55 | environment_response (growth_rate) | scalar | global | 3.2×10⁴ |
| EnvChemgenGirgis2009 | E. coli | Combined z-scores of MG1655 transposon mutants in 17 antibiotics. | 3,976 | 17 | 66,287 | environment_response (z_score) | scalar | global | 3.6×10⁷ |
| EnvChemgenNichols2011 | E. coli | Nichols 2011 Table S2: 3,979 E. coli strains' re-scaled scores in 324 screens. | 3,967 | 324 | 1,286,962 | environment_response (z_score) | scalar | global | 4.0×10⁸ |
| EnvChemgenShiver2016 | E. coli | Shiver 2016 KEIO deletion fitness-scores across 57 chemical-genomic conditions. | 3,846 | 51 | 210,998 | environment_response (z_score) | scalar | global | 9.9×10⁷ |
| EnvChemgenWang2015 | E. coli | Isoprenol tolerance of 46 Keio transporter deletions (Wang 2015, Table S3). | 46 | 1 | 46 | environment_response (log2_ratio) | scalar | global | 2.9×10⁴ |
| EnvChemgenWang2024 | E. coli | TRANSIT log2FC of MG1655 Tn5 mutants at three rifampicin doses x two times. | 4,419 | 6 | 26,514 | environment_response (log2_ratio) | scalar | global | 9.3×10⁶ |
| FitnessBrowserEcoli | E. coli | Fitness Browser Keio experiments not loaded from another release, by publication. | 3,789 | 3 | 22,734 | environment_response (log2_ratio) | scalar | global | 1.1×10⁷ |
| GrowthRateLamoureux2023 | E. coli | PRECISE-1K's released per-sample growth rate: 354 absolute rates in 1/hr. | 21 | 54 | 354 | environment_response (growth_rate) | scalar | global | 1.9×10⁵ |
| GrowthRateS23Schmidt2016 | E. coli | Schmidt 2016 Table S23: 24 absolute per-condition growth rates, three strains. | 1 | 22 | 26 | environment_response (growth_rate) | scalar | global | 2.0×10⁴ |
| GrowthYieldBaba2006 | E. coli | Baba 2006 Table 3: 11,735 released OD600 readings of obtained Keio mutants. | 3,912 | 3 | 11,735 | environment_response (optical_density) | scalar | global | 3.3×10⁶ |
| MismatchCrispriFitnessHawkins2020 | E. coli | Hawkins 2020 per-sgRNA relative fitness of E. coli essential-gene knockdowns. | 24,186 | 1 | 24,186 | environment_response (relative_growth_rate) | scalar | global | 1.5×10⁷ |
| PhageRbTnseqMutalik2020 | E. coli | Mutalik 2020 per-(gene, experiment) phage-challenge RB-TnSeq gene fitness. | 3,716 | 71 | 287,815 | environment_response (log2_ratio) | scalar | global | 2.1×10⁸ |
| RbTnseqPrice2018Ecoli | E. coli | Price 2018 compendium, E. coli BW25113 KEIO_ML9: gene fitness per sample. | 3,789 | 104 | 598,662 | environment_response (log2_ratio) | scalar | global | 4.0×10⁸ |
| RbTnseqTeteneva2024 | E. coli | Teteneva 2024, E. coli K-12 W3110 RpoS+ in lake water: Table S4 gene fitness. | 3,691 | 18 | 66,162 | environment_response (log2_ratio) | scalar | global | 2.3×10⁷ |
| RbTnseqWetmore2015 | E. coli | Wetmore 2015, E. coli BW25113 KEIO_ML9: gene fitness per successful sample. | 3,646 | 85 | 353,662 | environment_response (log2_ratio) | scalar | global | 2.4×10⁸ |
| EnvMetalTnseqRoyet2025 | P. putida | KT2440 mariner Tn-seq log2FC of every gene in cobalt, copper, zinc and cadmium. | 5,491 | 4 | 21,583 | environment_response (log2_ratio) | scalar | global | 6.7×10⁶ |
| FitnessBrowserPutida | P. putida | Fitness Browser KT2440 experiments of the attributed studies, by publication. | 9,556 | 146 | 1,118,052 | environment_response (log2_ratio) | scalar | global | 5.3×10⁸ |
| IsoprenolSelectionMenasalvas2025 | P. putida | Menasalvas 2025 enriched CRISPRi knockdowns from the growth-coupled selection. | 58 | 1 | 58 | environment_response (categorical) | scalar | global | 7.5×10⁴ |
| IsoprenolToleranceLim2025 | P. putida | Isoprenol tolerance of IPL300, IPL400 and KT2440 dPP_3024 (Lim 2025). | 3 | 2 | 3 | environment_response (log2_ratio) | scalar | global | 1.8×10⁴ |
| LactamGrowthRateThompson2019 | P. putida | Thompson 2019 Table S1: maximal growth rates on three carbon sources. | 3 | 3 | 9 | environment_response (growth_rate) | scalar | global | 1.2×10⁴ |
| RbTnseqBorchert2023 | P. putida | Borchert 2023 KT2440 RB-TnSeq fitness for the 271 loci the compendium dropped. | 271 | 32 | 10,824 | environment_response (log2_ratio) | scalar | global | 4.5×10⁶ |
| RbTnseqBorchert2024 | P. putida | KT2440 RB-TnSeq gene fitness, 4,732 genes x the kept samples of 332. | 9,464 | 155 | 1,466,920 | environment_response (log2_ratio) | scalar | global | 4.2×10⁸ |

#### Expression (RNA-seq)

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| MrnaFractionBalakrishnan2022 | E. coli | NCM3722-family mRNA number fractions, one record per released sample column. | 3 | 14 | 28 | mrna_number_fraction (rnaseq_mrna_number_fraction) | vector (4342) | node | 7.2×10⁶ |
| RnaseqCaglar2017 | E. coli | Caglar 2017 RNA-seq: one record per REL606 library (Table S2 sample). | 1 | 57 | 152 | expression_tpm (rnaseq_tpm) | vector (4196) | node | 6.7×10⁷ |
| RnaseqGuzman2018 | E. coli | Guzmán 2018's 14 released RNA-seq libraries, from its own GEO release. | 7 | 2 | 14 | expression_tpm (rnaseq_tpm) | vector (4329) | node | 4.5×10⁶ |
| RnaseqLamoureux2023 | E. coli | PRECISE-1K: one record per MG1655 RNA-seq library (wild type or deletions). | 74 | 119 | 1,035 | expression_tpm (rnaseq_tpm) | vector (4257) | node | 5.8×10⁸ |
| RnaseqPublicK12Lamoureux2023 | E. coli | Public K-12: one record per reprocessed public MG1655 RNA-seq library. | 72 | 198 | 1,675 | expression_tpm (rnaseq_tpm_from_released_counts) | vector (4355) | node | 9.5×10⁸ |
| PutidaPrecise321Lim2022 | P. putida | putidaPRECISE321: per-sample absolute KT2440 transcriptomes (TPM + counts). | 14 | 118 | 321 | expression_tpm (rnaseq_tpm) | vector (5564) | node | 2.4×10⁸ |

#### Fitness

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| CarbonSourceTong2020 | E. coli | Tong 2020 Keio and sRNA/small protein deletion growth on 30 carbon sources. | 3,714 | 30 | 111,420 | fitness | scalar | global | 4.8×10⁷ |
| CrispriCrossRachwalski2024 | E. coli | Rachwalski 2024 normalized colony growth of CRISPRi knockdown by deletion crosses. | 14,209 | 15 | 48,900 | fitness | scalar | global | 1.9×10⁷ |
| GrowthAucRapp2026 | E. coli | Growth of Rapp 2026's CRISPRi library, as the paper's AUC over the controls. | 1,515 | 1 | 1,515 | fitness | scalar | global | 7.7×10⁵ |
| GrowthRateCampos2018 | E. coli | Campos 2018 Keio maximal-growth-rate fitness in M9 casamino-acid glucose. | 3,664 | 1 | 3,664 | fitness | scalar | global | 1.1×10⁶ |
| GrowthRateChoe2019 | E. coli | Choe 2019 Fig. 2c: the two MS56 designed deletions, as parent-relative fitness. | 6 | 1 | 8 | fitness | scalar | global | 1.1×10⁴ |
| GrowthRateSchmidt2016 | E. coli | Schmidt 2016 Table S24: rim-deletion growth rates as WT-relative fitness. | 3 | 2 | 6 | fitness | scalar | global | 7.8×10³ |
| GrowthWang2015 | E. coli | Plain-medium growth of 46 Keio transporter deletions (Wang 2015, Table S3). | 46 | 1 | 46 | fitness | scalar | global | 2.1×10⁴ |
| TranscriptionFactorKnockoutChoe2019 | E. coli | Choe 2019 Supplementary Fig. 6: the two Keio deletions that carry a number. | 2 | 1 | 2 | fitness | scalar | global | 5.3×10³ |
| FitnessContrastBorchert2023 | P. putida | Borchert 2023's 64,853 released two-arm fitness contrasts. | 5,031 | 12 | 64,853 | t_statistic (rbtnseq_gene_fitness_two_sample_t_test) | scalar | global | 2.2×10⁷ |

#### Flux

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| FluxIshii2007 | E. coli | Fitted 13C net-flux maps of 28 chemostat cultures. | 25 | 5 | 28 | net_flux (c13_mfa_net_flux_percent_of_specific_glucose_uptake) | vector (42) | bipartite node | 6.0×10⁴ |
| FluxLi2021 | E. coli | Li 2021's three released 13C-MFA maps, against BW-P BF's unreleased map. | 3 | 1 | 3 | net_flux (c13_mfa_net_flux_mmol_per_gdw_per_h) | vector (99) | bipartite node | 5.7×10⁴ |

#### Gene interaction

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| GeneInteractionBabu2014 | E. coli | eSGA interaction scores of E. coli K-12 double deletion mutants (Table S2). | 42,703 | 1 | 42,705 | gene_interaction | scalar | edge | 1.2×10⁷ |
| GeneInteractionButland2008 | E. coli | Unfiltered eSGA interaction scores of E. coli K-12 double deletion mutants. | 314,769 | 1 | 314,769 | gene_interaction | scalar | edge | 8.2×10⁷ |

#### Metabolite

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ExometaboliteBrunk2016 | E. coli | The six HPLC columns the workbook labels g/L: glucose and five acids. | 9 | 28 | 126 | metabolite_level (hplc_extracellular_concentration_g_per_l) | vector (6) | bipartite node | 1.8×10⁵ |
| MetaboliteIntensityRapp2026 | E. coli | Absolute FI-MS intensity of Rapp 2026's accumulating annotated features. | 407 | 1 | 407 | metabolite_level (fi_ms_annotated_feature_absolute_intensity_mean_of_2_plates) | vector (2) | bipartite node | 3.9×10⁵ |
| MetabolomeBrunk2016 | E. coli | The 51 LC-MS columns the workbook labels (uM), one record per sample. | 9 | 26 | 117 | metabolite_level (lc_ms_concentration_um) | vector (51) | bipartite node | 3.5×10⁵ |
| MetabolomeFuhrer2017 | E. coli | Genome-wide FIA-TOF-MS ion metabolome of the E. coli Keio deletion collection. | 3,806 | 1 | 3,806 | metabolite_level (fia_tof_ms_ion_modified_z_score) | vector (7534) | bipartite node | 1.9×10⁹ |
| MetabolomeIshii2007 | E. coli | CE-TOFMS intracellular metabolite concentrations of 28 chemostat cultures. | 25 | 5 | 28 | metabolite_level (ce_tofms_intracellular_concentration_mm) | vector (130) | bipartite node | 4.9×10⁵ |
| MetabolomeRapp2026 | E. coli | FI-MS metabolome of a CRISPRi library covering every iML1515 gene of E. coli. | 1,497 | 1 | 1,497 | metabolite_level (fi_ms_iml1515_feature_fold_change_vs_batch_median_mean_of_2_plates) | vector (1321) | bipartite node | 5.6×10⁸ |
| MetabolomeSchastnaya2021 | E. coli | FIA-TOF-MS metabolome of the KEIO deletion arm of an E. coli phosphosite screen. | 103 | 5 | 200 | metabolite_level (fia_tof_ms_ion_log2_fold_change_vs_wild_type) | vector (284) | bipartite node | 3.6×10⁶ |
| TargetedMetabolomeRapp2026 | E. coli | Targeted LC-MS/MS of Rapp 2026's accumulating strain-metabolite pairs. | 407 | 1 | 407 | metabolite_level (lc_msms_eic_peak_height_fold_change_vs_library_median) | vector (2) | bipartite node | 3.7×10⁵ |
| MetaboliteGrowthPhaseMenasalvas2025 | P. putida | data S1-1's growth-phase (GP) absolute metabolite concentrations. | 3 | 1 | 3 | metabolite_level (lc_ms_intracellular_concentration_uM) | vector (46) | bipartite node | 7.4×10⁴ |
| MetaboliteProductionPhaseMenasalvas2025 | P. putida | data S1-1's production-phase (PP) absolute metabolite concentrations. | 3 | 1 | 3 | metabolite_level (lc_ms_intracellular_concentration_uM) | vector (46) | bipartite node | 7.3×10⁴ |

#### Morphology

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| MorphologyCampos2018 | E. coli | Campos 2018 Keio cell-morphology profiles: 26 features per imaged strain. | 3,664 | 1 | 3,664 | morphology | vector (15) | global | 8.5×10⁶ |

#### Phosphoproteome

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| PhosphositeIntensityPotel2018 | E. coli | Potel 2018 phosphosite intensities of W3110 in 12 released samples. | 1 | 2 | 12 | phosphosite_intensity (fe3_imac_lc_msms_maxquant_summed_xic) | vector (989) | node | 1.3×10⁷ |

#### Product titer

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| BiofuelTiterBrunk2016 | E. coli | Brunk 2016 isopentenol, limonene and bisabolene titers over the time course. | 8 | 18 | 72 | titer | scalar | global | 1.1×10⁵ |
| IsopentenolTiterFoo2014 | E. coli | Foo 2014 isopentenol titers: the eight tolerance strains and the RFP control. | 9 | 1 | 9 | titer | scalar | global | 1.5×10⁴ |
| PineneFoldChangeNiu2019 | E. coli | Niu 2019 CRISPRa/CRISPRi pinene production ratios in BW25113(PT5-dxs). | 40 | 1 | 40 | titer | scalar | global | 2.2×10⁴ |
| IsoprenolTiterBanerjee2024 | P. putida | Banerjee 2024 isoprenol titers: the 20 the article body states numerically. | 14 | 9 | 20 | titer | scalar | global | 3.4×10⁴ |
| IsoprenolTiterCarruthers2025 | P. putida | Carruthers 2025 per-strain isoprenol titer across six DBTL cycles. | 432 | 2 | 502 | titer | scalar | global | 4.2×10⁵ |
| IsoprenolTiterDeSiqueira2025 | P. putida | de Siqueira 2025 maximum isoprenol titer of PT carrying pIY670. | 5 | 2 | 10 | titer | scalar | global | 2.1×10⁵ |
| IsoprenolTiterYunus2026 | P. putida | Yunus 2026 per-strain isoprenol titer, from the hand-deposited Benchling table. | 125 | 1 | 125 | titer | scalar | global | 6.6×10⁴ |
| IsoprenylAcetateTiterKang2026 | P. putida | Kang 2026 isoprenyl acetate titers: Table 1, Table S4 and the Table S9 fed-batch. | 11 | 13 | 26 | titer | scalar | global | 6.7×10⁴ |
| ValerolactamTiterThompson2019 | P. putida | Thompson 2019 valerolactam titers: the four-strain ladder at 24 and 48 h. | 4 | 2 | 8 | titer | scalar | global | 1.7×10⁴ |

#### Promoter activity

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| PromoterReporterMohiuddin2022 | E. coli | Mohiuddin 2022 hourly promoter-GFP readings under three antibiotics. | 1,809 | 27 | 69,480 | promoter_activity | scalar | node | 2.8×10⁷ |

#### Protein abundance

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ProteinFoldChangeCaglar2017 | E. coli | Caglar 2017 differential proteomics: one record per Table S8 protein contrast. | 1 | 8 | 8 | protein_fold_change (deseq2_wald_log2_fold_change_design_batch_plus_condition) | vector (4180) | node | 1.1×10⁷ |
| ProteomeBrunk2016 | E. coli | Brunk 2016 SRM peak areas: nine strains at nine hours, all 68 host proteins. | 9 | 18 | 81 | protein_abundance (srm_protein_peak_area) | vector (68) | node | 3.6×10⁵ |
| ProteomeCaglar2017 | E. coli | Caglar 2017 proteome: one record per REL606 LC-MS/MS sample (Table S3). | 1 | 40 | 105 | protein_abundance (lcmsms_spectral_count_deseq2_size_factor_normalized) | vector (4196) | node | 2.5×10⁷ |
| ProteomeIshii2007 | E. coli | LC-MS/MS absolute protein abundances of 28 chemostat cultures. | 25 | 5 | 28 | protein_abundance (lc_ms_ms_absolute_mg_protein_per_g_dry_cell_weight) | vector (57) | node | 2.9×10⁵ |
| ProteomeMori2021 | E. coli | Absolute MG1655 proteome mass fractions, one record per loaded sample. | 1 | 1 | 7 | protein_abundance (absolute_protein_mass_fraction_xtop_dia_swath_riboprofiling_scaled) | vector (1897) | node | 1.1×10⁶ |
| ProteomeSchmidt2016 | E. coli | Absolute BW25113 proteome, one record per loaded growth condition. | 1 | 21 | 21 | protein_abundance (absolute_protein_copies_per_cell_label_free_sid_anchored) | vector (2359) | node | 1.1×10⁷ |
| ProteomeSrmSet1Schmidt2016 | E. coli | Table S2: the OGE-fractionated arm's SRM absolute abundances, 11 records. | 1 | 18 | 18 | protein_abundance (absolute_protein_copies_per_cell_srm_sid_dataset1_technical_sd) | vector (41) | node | 1.5×10⁵ |
| ProteomeSrmSet2Schmidt2016 | E. coli | Table S3: the unfractionated arm's SRM absolute abundances, 14 records. | 1 | 21 | 21 | protein_abundance (absolute_protein_copies_per_cell_srm_sid_dataset2_biological_sd) | vector (31) | node | 1.3×10⁵ |
| CampaignProteomeCarruthers2025 | P. putida | Carruthers 2025 campaign proteome: one record per CRISPRi strain-cycle. | 401 | 1 | 465 | protein_abundance (dia_nn_top3_percent_of_proteome_mean) | vector (1616) | node | 1.3×10⁸ |
| CrispriArrayYunus2026 | P. putida | Yunus 2026 multiplexed-array relative expression panels (Tables S8-S12). | 25 | 1 | 25 | protein_fold_change (dia_nn_top3_relative_to_control_strain) | scalar | node | 2.5×10⁴ |
| CrispriDifferentialProteomeYunus2026 | P. putida | Yunus 2026 global fold change against the control for the PP_4188 strain. | 1 | 1 | 1 | protein_fold_change (dia_nn_top3_fold_change_relative_to_control_strain_equal_variance_t_test) | vector (305) | node | 6.1×10⁴ |
| CrispriKnockdownYunus2026 | P. putida | Yunus 2026 per-strain relative expression of its own CRISPRi target (Table S3). | 102 | 1 | 102 | protein_fold_change (dia_nn_top3_relative_to_control_strain) | scalar | node | 4.9×10⁴ |
| CrispriPanelProteomeYunus2026 | P. putida | Yunus 2026 per-strain Top3 protein panel, from the hand-deposited table. | 125 | 1 | 125 | protein_abundance (dia_nn_top3_signal_benchling_displayed) | vector (207) | node | 1.5×10⁶ |
| ProteomeBanerjee2025 | P. putida | Banerjee 2025 DIA proteome: six released samples of the p-CA cutset strains. | 4 | 3 | 6 | protein_abundance (diann_lfq_log2_intensity_replicate_mean) | vector (2248) | node | 2.8×10⁶ |
| ProteomeCarruthers2025 | P. putida | Carruthers 2025 Top3 proteome of the PP_0815 off-target-study strains. | 17 | 2 | 21 | protein_abundance (dia_nn_top3_peptide_signal_mean) | vector (1424) | node | 5.6×10⁶ |
| ProteomeDeSiqueira2025 | P. putida | de Siqueira 2025 Top3 proteome of the writable KT2440 and PT strains. | 6 | 3 | 17 | protein_abundance (dia_nn_top3_peptide_signal_replicate_mean) | vector (1531) | node | 5.5×10⁶ |
| ProteomeFoldChangeCarruthers2025 | P. putida | Carruthers 2025 per-protein log2 fold changes of the CRISPRi and KO contrasts. | 16 | 1 | 16 | protein_fold_change (dia_log2_fold_change_paired_two_tailed_t_test) | vector (350) | node | 1.5×10⁶ |
| ProteomeFoldChangeLim2025 | P. putida | Lim 2025's four released evolved-isolate-over-IPL400 proteome contrasts. | 2 | 2 | 4 | protein_fold_change (dia_nn_top3_log2_fold_change_welch_two_sample_t_test) | vector (2361) | node | 3.7×10⁶ |
| ProteomeLim2025 | P. putida | IPL400 and its two evolved isolates under 4 g/L isoprenol (Lim 2025). | 3 | 1 | 3 | protein_abundance (dia_nn_top3_log2_mean) | vector (2361) | node | 1.5×10⁶ |
| ProteomeLog10PercentDeSiqueira2025 | P. putida | de Siqueira 2025 mean log10 percent-of-total abundance, the third released scale. | 6 | 3 | 17 | protein_abundance (dia_nn_top3_log10_percent_of_total_abundance_replicate_mean) | vector (1531) | node | 6.0×10⁶ |
| ProteomeMenasalvas2025 | P. putida | Menasalvas 2025 growth-phase and production-phase proteome of the three strains. | 3 | 2 | 6 | protein_abundance (dia_protein_counts_sum_mean) | vector (2090) | node | 2.6×10⁶ |
| ProteomePercentDeSiqueira2025 | P. putida | de Siqueira 2025 percent-of-total relative abundance, the paper's own scale. | 6 | 3 | 17 | protein_abundance (dia_nn_top3_percent_of_total_abundance_replicate_mean) | vector (1531) | node | 6.0×10⁶ |

#### Protein synthesis / turnover

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ProteinSynthesisRateLi2014 | E. coli | Li 2014 absolute protein synthesis rates of MG1655 in three MOPS media. | 1 | 3 | 3 | synthesis_rate (ribosome_profiling_footprint_density_absolute_synthesis_rate) | vector (3041) | node | 3.7×10⁵ |
| ProteinTurnoverGupta2024 | E. coli | Gupta 2024 per-protein total turnover in 13 NCM3722 growth conditions. | 6 | 8 | 13 | degradation_rate (n15_ammonium_tmtproc_total_turnover_rate_per_hour) | vector (2553) | node | 1.1×10⁷ |

#### Viability

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| GeneEssentialityBaba2006 | E. coli | Keio construction outcomes of 4,288 targeted E. coli BW25113 ORFs. | 4,288 | 1 | 4,288 | is_essential | scalar | node | 1.8×10⁶ |
| GeneEssentialityGerdes2003 | E. coli | Genetic-footprinting essentiality calls of E. coli MG1655 genes (Table S1). | 3,739 | 1 | 3,739 | is_essential | scalar | node | 1.9×10⁶ |
| GeneEssentialityGoodall2018 | E. coli | TraDIS essentiality calls of E. coli BW25113 genes, as plated and after LB. | 8,309 | 2 | 8,309 | is_essential | scalar | node | 5.1×10⁶ |
| GeneEssentialityPrice2018Ecoli | E. coli | Price 2018 Supplementary Table 1: E. coli BW25113 likely-essential genes. | 324 | 1 | 324 | is_essential | scalar | node | 2.6×10⁵ |
| GradedGeneEssentialityGoodall2018 | E. coli | Goodall 2018's 317 "unclear" TraDIS calls, stored as released (#909). | 317 | 2 | 317 | essentiality_call | scalar | node | 2.2×10⁵ |
| GradedGeneEssentialityRoyet2025 | P. putida | Royet 2025 Table S4: every KT2440 gene's TRANSIT HMM state on LB agar and after outgrowth in LB, stored as released (#909). | 5,729 | 2 | 11,458 | essentiality_call | scalar | node | 4.8×10⁶ |

#### Total

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| **Total (100 datasets, 100 probed)** |  |  |  |  | **7,257,468** |  |  |  | **7.7×10⁹** |

Source JSON: `experiments/database/results/pre-build/2026-10-15/ecoli-pputida_datasets.json`.
