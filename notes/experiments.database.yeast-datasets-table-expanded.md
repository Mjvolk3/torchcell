---
id: ry0k1snhsftlk451nvygye6
title: Yeast Datasets Table Expanded
desc: ''
updated: 1792110900029
created: 1792110900029
---

## Yeast datasets, expanded: the main table with a description of each dataset

<!-- GENERATED: raw data from experiments/database/scripts/build_bacteria_datasets_table.py
(-> results/pre-build/<date>/<organisms>_datasets.json); this note is a VIEW rendered by
render_bacteria_datasets_table.py. Do not edit tables by hand. -->

The table of [[experiments.database.yeast-datasets-table]] with one more column.
*Description* is the summary paragraph of the dataset class's docstring in
`torchcell/datasets/`, attached by the build script when it writes the JSON, so a better
description is made by improving that docstring and rerunning both scripts. Every other
column is read from the built LMDB; definitions and the Signal controls are in the
non-expanded note. Commentary: [[experiments.database.scripts.build_bacteria_datasets_table]].

### Datasets

#### Environmental / chemogenomic response

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| CellYieldDeChiara2022 | S. cerevisiae | Colony cell yield (cells) of 1,011 isolates in 43 environments. | 906 | 43 | 36,286 | environment_response (cell_yield) | scalar | global | 1.7×10⁷ |
| CrisprMagicLian2019 | S. cerevisiae | Lian 2019 MAGIC per-guide CRISPRa/i/d furfural-enrichment env x geno dataset. | 266,304 | 3 | 266,304 | environment_response (log2_ratio) | scalar | global | 1.3×10⁸ |
| CrispriChemgenSmith2016 | S. cerevisiae | Smith 2016 per-guide dCas9-Mxi1 CRISPRi chemical-genetic env x geno dataset. | 1,035 | 26 | 14,463 | environment_response (log2_ratio) | scalar | global | 7.5×10⁶ |
| CrispriMormino2022 | S. cerevisiae | Mormino 2022 CRISPRi acetic-acid biosensor screen (12 isolated strains). | 12 | 1 | 12 | environment_response (categorical) | scalar | global | 1.4×10⁴ |
| DoublingTimeDeChiara2022 | S. cerevisiae | Colony population doubling time (h) of 1,011 isolates in 43 environments. | 906 | 43 | 36,295 | environment_response (growth_rate) | scalar | global | 1.7×10⁷ |
| EnvChemgenAuesukaree2009 | S. cerevisiae | Auesukaree 2009 six-stress env x geno -> categorical sensitivity dataset. | 333 | 6 | 525 | environment_response (categorical) | scalar | global | 3.1×10⁵ |
| EnvChemgenCostanzo2021 | S. cerevisiae | Costanzo 2021 condition-SGA: env x (deletion \| ts-allele) -> differential fitness. | 4,422 | 14 | 61,640 | environment_response (differential_fitness) | scalar | global | 1.1×10⁸ |
| EnvChemgenHoepfner2014 | S. cerevisiae | Hoepfner 2014 HIP-HOP atlas: env x (het \| hom kanMX4 deletion) -> sensitivity score. | 10,839 | 5,586 | 30,308,574 | environment_response (sensitivity_score) | scalar | global | 2.1×10¹⁰ |
| EnvChemgenMota2024 | S. cerevisiae | Acetic/butyric/octanoic acid chemogenomic env x geno -> ordinal susceptibility. | 607 | 3 | 1,289 | environment_response (ordinal) | scalar | global | 7.4×10⁵ |
| EnvChemgenParsons2006 | S. cerevisiae | Parsons 2006 chemical-genetic compendium: env x geno -> log2(control/treated). | 4,111 | 82 | 329,254 | environment_response (log2_ratio) | scalar | global | 2.2×10⁸ |
| EnvChemgenPiotrowski2017 | S. cerevisiae | MOSAIC diagnostic-pool chemical genetics: env x geno -> CG interaction z-score. | 289 | 11,949 | 3,435,820 | environment_response (z_score) | scalar | global | 1.4×10⁹ |
| EnvChemgenVanacloig2022 | S. cerevisiae | Anaerobic chemical-genomic env x geno -> log2(inhibitor/control) response screen. | 3,649 | 45 | 164,205 | environment_response (log2_ratio) | scalar | global | 9.3×10⁷ |
| EnvChemgenWildenhain2015 | S. cerevisiae | Wildenhain 2015 chemical-genetic matrix: env x geno -> growth-inhibition z-score. | 258 | 5,178 | 449,033 | environment_response (z_score) | scalar | global | 2.6×10⁸ |
| EnvChemgenYoshikawa2009 | S. cerevisiae | Yoshikawa 2009 growth rates: env x geno -> one culture's specific growth rate. | 4,729 | 4 | 37,832 | environment_response (growth_rate) | scalar | global | 1.0×10⁷ |
| FattyAcidSmith2006 | S. cerevisiae | Smith 2006 fatty-acid clear-zone env x geno -> ordinal response dataset. | 4,770 | 3 | 14,310 | environment_response (ordinal) | scalar | global | 1.8×10⁶ |
| HetHillenmeyer2008 | S. cerevisiae | FitDb HIP (heterozygous kanMX4 allele replacement) fitness-defect log2-ratio. | 17,659 | 512 | 3,074,132 | environment_response (log2_ratio) | scalar | global | 2.0×10⁹ |
| HomHillenmeyer2008 | S. cerevisiae | FitDb HOP (homozygous barcoded kanMX4 deletion) fitness-defect z-score dataset. | 9,220 | 276 | 1,228,676 | environment_response (z_score) | scalar | global | 8.4×10⁸ |
| IsolateGrowthPeter2018 | S. cerevisiae | Peter 2018: one record per (isolate, condition), 33,985 records. | 971 | 35 | 33,985 | environment_response (condition_growth_ratio) | scalar | global | 1.6×10⁷ |

#### Expression (RNA-seq)

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Boocock2025 | S. cerevisiae | Boocock 2025: 393 pooled segregants plus 79,123 de novo segregant cells. | 79,516 | 2 | 79,516 | umi_count (scrnaseq_umi_count) | vector (3797) | node | 8.9×10⁹ |
| CaudalPanTranscriptome2024 | S. cerevisiae | Natural-isolate pan-transcriptome: each isolate a perturbation set off S288C. | 943 | 1 | 943 | expression_tpm (rnaseq_tpm) | vector (6017) | node | 1.7×10⁹ |
| NadalRibellesPerturbSeq2025 | S. cerevisiae | Nadal-Ribelles 2025 pseudobulk Perturb-seq: one record per (genotype, condition). | 3,150 | 2 | 6,188 | expression_log2_ratio (pseudobulk_scrnaseq_log2fc) | vector (5818) | node | 2.5×10⁹ |
| Nguessan2025 | S. cerevisiae | N'Guessan 2025: 2,951 BB-QTL batch-1 segregants, their cells summed. | 2,951 | 1 | 2,951 | umi_count (scrnaseq_umi_count) | vector (1670) | node | 4.5×10⁸ |
| PooledPromoterExpressionVaishnav2022 | S. cerevisiae | Vaishnav 2022 GPRA sort-seq expression of 71,158 promoter sequences in SC-Ura. | 71,158 | 1 | 71,158 | expression_level | scalar | node | 1.3×10⁸ |
| TfDeletionPseudobulkJackson2020 | S. cerevisiae | Jackson 2020: one record per (barcoded strain, condition), 784 records. | 72 | 11 | 784 | umi_count (scrnaseq_umi_count) | vector (5075) | node | 9.7×10⁷ |

#### Expression (microarray)

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| DmMicroarraySameith2015 | S. cerevisiae | Double mutant GSTF expression profiling from Sameith et al. 2015. | 72 | 1 | 72 | expression_log2_ratio | vector (6169) | node | 1.7×10⁸ |
| InductionTimecourseHackett2020 | S. cerevisiae | Hackett 2020 IDEA: one record per (time course, time point), 1,752 records. | 214 | 31 | 1,752 | log2_cleaned | vector (6175) | node | 4.7×10⁹ |
| MicroarrayKemmeren2014 | S. cerevisiae | Deletion-mutant microarray expression profiling from Kemmeren et al. 2014. | 1,484 | 1 | 1,484 | expression_log2_ratio | vector (6169) | node | 3.2×10⁹ |
| MicroarrayScoreHu2007 | S. cerevisiae | Hu 2007 TF-deletion transcriptomes as released: error-model X scores + p. | 263 | 1 | 263 | expression_score (error_model_x_score_weighted_duplicate_average) | vector (6371) | node | 2.5×10⁸ |
| SmMicroarraySameith2015 | S. cerevisiae | Single mutant GSTF expression profiling from Sameith et al. 2015. | 82 | 1 | 82 | expression_log2_ratio | vector (6169) | node | 1.8×10⁸ |

#### Fitness

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Bloom2019 | S. cerevisiae | Bloom 2019: one record per (segregant, condition), 530,100 records. | 13,950 | 38 | 530,100 | environment_response (control_regression_residual) | scalar | global | 2.6×10⁸ |
| CrispriSegregantHale2024 | S. cerevisiae | Hale 2024: one record per (double-barcode lineage, assay), 8,692,635 records. | 895,494 | 2 | 8,692,635 | environment_response (relative_fitness) | scalar | global | 6.7×10⁹ |
| DmfCostanzo2016 | S. cerevisiae | Double-mutant fitness dataset from Costanzo et al. 2016. | 20,705,612 | 2 | 20,705,612 | fitness | scalar | global | 7.1×10⁹ |
| DmfKuzmin2018 | S. cerevisiae | Double-mutant fitness experiments from Kuzmin 2018. | 410,571 | 1 | 410,571 | fitness | scalar | global | 1.5×10⁸ |
| DmfKuzmin2020 | S. cerevisiae | Double-mutant fitness experiments from Kuzmin et al. 2020. | 608,805 | 1 | 632,998 | fitness | scalar | global | 2.4×10⁸ |
| SmfBaryshnikova2010 | S. cerevisiae | Baryshnikova 2010 genome-scale single-mutant fitness dataset (6023 alleles). | 6,023 | 2 | 6,023 | fitness | scalar | global | 1.8×10⁶ |
| SmfCostanzo2016 | S. cerevisiae | Single-mutant fitness dataset from Costanzo et al. 2016. | 10,277 | 2 | 20,484 | fitness | scalar | global | 5.4×10⁶ |
| SmfKuzmin2018 | S. cerevisiae | Single-mutant fitness experiments from Kuzmin 2018. | 1,539 | 1 | 1,539 | fitness | scalar | global | 3.9×10⁵ |
| SmfKuzmin2020 | S. cerevisiae | Single-mutant fitness experiments from Kuzmin et al. 2020. | 472 | 1 | 472 | fitness | scalar | global | 9.3×10⁴ |
| SmfODuibhir2014 | S. cerevisiae | O'Duibhir 2014 single-deletion relative-growth-rate -> fitness dataset. | 1,312 | 1 | 1,312 | fitness | scalar | global | 2.0×10⁶ |
| TmfKuzmin2018 | S. cerevisiae | Triple-mutant fitness experiments from Kuzmin 2018. | 91,111 | 1 | 91,111 | fitness | scalar | global | 3.9×10⁷ |
| TmfKuzmin2020 | S. cerevisiae | Triple-mutant fitness experiments from Kuzmin et al. 2020. | 290,355 | 1 | 301,798 | fitness | scalar | global | 1.3×10⁸ |

#### Gene interaction

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| DmiCostanzo2016 | S. cerevisiae | Double-mutant genetic interaction dataset from Costanzo et al. 2016. | 20,705,612 | 2 | 20,705,612 | gene_interaction | scalar | edge | 6.0×10⁹ |
| DmiKuzmin2018 | S. cerevisiae | Double-mutant genetic-interaction experiments from Kuzmin 2018. | 410,399 | 1 | 410,399 | gene_interaction | scalar | edge | 1.3×10⁸ |
| DmiKuzmin2020 | S. cerevisiae | Digenic interaction experiments from Kuzmin et al. 2020. | 608,604 | 1 | 632,797 | gene_interaction | scalar | edge | 2.1×10⁸ |
| TmiKuzmin2018 | S. cerevisiae | Triple-mutant genetic-interaction experiments from Kuzmin 2018. | 91,111 | 1 | 91,111 | gene_interaction | scalar | hyperedge | 3.4×10⁷ |
| TmiKuzmin2020 | S. cerevisiae | Trigenic interaction experiments from Kuzmin et al. 2020. | 290,355 | 1 | 301,798 | gene_interaction | scalar | hyperedge | 1.1×10⁸ |

#### Genome architecture

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| GenomeArchitecturePuddu2019 | S. cerevisiae | Puddu 2019: one record per sequenced knockout colony, 8,835 records. | 4,550 | 1 | 8,835 | genome_architecture | vector (28) | global | 1.2×10⁷ |

#### Life cycle

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| LifeCycleDeChiara2022 | S. cerevisiae | Sporulation and stationary-phase survival time courses of 1,011 isolates. | 905 | 5 | 4,039 | life_cycle | vector (4) | global | 1.3×10⁶ |

#### Metabolite

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| AminoAcidCooper2010 | S. cerevisiae | Cooper 2010 CE-LIF amino-acid pools of the deletion collection (ratio to plate). | 4,334 | 1 | 4,382 | metabolite_level (ce_lif_peak_area_ratio_to_plate_mean) | vector (16) | bipartite node | 5.2×10⁶ |
| AminoAcidMulleder2016 | S. cerevisiae | Genome-wide intracellular amino-acid metabolome of the yeast deletion collection. | 4,678 | 1 | 4,678 | metabolite_level (intracellular_concentration_mM) | vector (19) | bipartite node | 1.1×10⁷ |
| BetaxanthinCachera2023 | S. cerevisiae | Genome-wide betaxanthin CRI-SPA product-proxy screen of the YKO collection. | 4,759 | 1 | 4,759 | metabolite_level (cri_spa_corrected_hsv_yellowness_24h) | scalar | bipartite node | 4.5×10⁶ |
| FattyAcidXue2025 | S. cerevisiae | Xue 2025 in-house FFA combinatorial-deletion titer dataset (5 FFAs/strain). | 176 | 1 | 176 | metabolite_level (titer_mg_per_l) | vector (5) | bipartite node | 1.9×10⁵ |
| IsobutanolScreenLopez2024 | S. cerevisiae | Montaño López 2024 FIRST genome-wide isobutanol-biosensor YKO screen (Table S2, n=1). | 4,607 | 1 | 4,805 | metabolite_level (biosensor_gfp_fluorescence_fold_change) | scalar | bipartite node | 9.0×10⁵ |
| IsobutanolValidatedLopez2024 | S. cerevisiae | Montaño López 2024 VALIDATED isobutanol-biosensor re-screen (Table S3, triplicate). | 226 | 1 | 227 | metabolite_level (biosensor_gfp_fluorescence_fold_change) | scalar | bipartite node | 7.4×10⁴ |
| MetaboliteDaSilveira2014 | S. cerevisiae | MS lipidome of the yeast kinase/phosphatase deletion collection (127 mutants). | 127 | 1 | 127 | metabolite_level (lipidomics_ms_relative_abundance_au) | vector (135) | bipartite node | 1.1×10⁶ |
| MetaboliteZelezniak2018 | S. cerevisiae | LC-SRM targeted metabolome of the kinase-knockout collection, one record per (strain, protocol); each record and its WT reference share one protocol. | 95 | 1 | 129 | metabolite_level (lc_srm_signal_batch_corrected_uncalibrated_dataset1) | vector (14) | bipartite node | 3.5×10⁵ |
| OrganicAcidYoshida2012 | S. cerevisiae | HPLC organic-acid titers of 17 yeast deletion mutants (Yoshida 2012, Table 3). | 17 | 1 | 17 | metabolite_level (hplc_organic_acid_titer_mM) | vector (6) | bipartite node | 3.3×10⁴ |

#### Morphology

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| CarotenoidOzaydin2013 | S. cerevisiae | Beta-carotene colony-color visual screen of the yeast deletion collection. | 4,475 | 1 | 4,607 | visual_score | scalar | global | 7.7×10⁶ |
| ScmdOhnuki2018 | S. cerevisiae | CalMorph morphology of essential-gene heterozygous diploid deletions (Ohnuki 2018). | 1,112 | 1 | 1,112 | calmorph | vector (281) | global | 4.9×10⁷ |
| ScmdOhnuki2022 | S. cerevisiae | CalMorph morphology for 1979 quadruple (target + 3Delta) deletion strains. | 1,982 | 1 | 1,982 | calmorph | vector (281) | global | 8.8×10⁷ |
| ScmdOhya2005 | S. cerevisiae | CalMorph morphology for haploid non-essential single-gene deletions (Ohya 2005). | 4,718 | 1 | 4,718 | calmorph | vector (281) | global | 1.5×10⁸ |

#### Protein abundance

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ProteomeMessner2021 | S. cerevisiae | Scanning SWATH proteomes of wild-type yeast under 18 compounds vs DMSO (70 runs). | 1 | 18 | 70 | protein_abundance (swath_ms_diann_maxlfq_proteotypic_quantity) | vector (1920) | node | 1.2×10⁷ |
| ProteomeMessner2023 | S. cerevisiae | Microflow-SWATH-MS proteome of the yeast gene-KO collection (4,699 strains). | 4,549 | 1 | 4,699 | protein_abundance (swath_ms_maxlfq_batch_corrected_quantity) | vector (1830) | node | 1.2×10⁹ |
| ProteomeMuenzner2024 | S. cerevisiae | 796 SWATH-MS proteomes (SI Table 3): 794 isolates with a Peter genome and FY4. | 795 | 1 | 795 | protein_abundance (swath_ms_maxlfq_precursor_batch_corrected_quantity) | vector (1542) | node | 9.6×10⁸ |
| ProteomeZelezniak2018 | S. cerevisiae | SWATH-MS proteome of the yeast kinase-knockout collection (97 strains). | 97 | 1 | 97 | protein_abundance (swath_ms_label_free_log_signal_sva) | vector (726) | node | 1.7×10⁷ |

#### Protein synthesis / turnover

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| ProteinTurnoverMuenzner2024 | S. cerevisiae | Dynamic-SILAC turnover rates for 55 natural isolates (SI Table 17). | 55 | 1 | 55 | degradation_rate (dynamic_silac_lys8_kdp_per_hour_no_dilution_correction) | vector (1015) | node | 6.9×10⁷ |

#### Viability

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| GeneEssentialitySgd | S. cerevisiae | Gene essentiality experiments built from SGD inviable null phenotypes. | 1,180 | 5 | 1,329 | is_essential | scalar | node | 6.8×10⁵ |
| SynthLethalityYeastSynthLethDb | S. cerevisiae | Yeast synthetic lethality gene-pair experiments from SynLethDB. | 14,000 | 1 | 14,000 | is_synthetic_lethal | scalar | edge | 8.0×10⁶ |
| SynthRescueYeastSynthLethDb | S. cerevisiae | Yeast synthetic rescue gene-pair experiments from SynLethDB. | 6,948 | 1 | 6,948 | is_synthetic_rescue | scalar | edge | 4.0×10⁶ |

#### Total

| Dataset | Host | Description | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| **Total (69 datasets, 69 probed)** |  |  |  |  | **93,266,786** |  |  |  | **7.2×10¹⁰** |

Source JSON: `experiments/database/results/pre-build/2026-10-15/scerevisiae_datasets.json`.
