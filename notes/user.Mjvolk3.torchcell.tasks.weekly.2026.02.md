---
id: 7f5qxqvqi4si81oyv3p39y9
title: '02'
desc: ''
updated: 1768415096062
created: 1767743671512
---

## 2026.01.06

- [x] notes on [[Triple_interaction_enrichment_of_uncharacterized_genes|experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes]]
- [x] 010 inference scripts for 3 models (m00, m01, m02) - renamed from E006/r0 naming
- [x] 12/24 gene panel selection for wetlab with best model [[select_12_and_24_genes_top_triples|experiments.010-kuzmin-tmi.scripts.select_12_and_24_genes_top_triples]]
- [x] panel histogram visualizations (C(12,3)=220, C(24,3)=2024 constructible triples) [[inferred_triple_interaction_histogram_from_gene_selection|experiments.010-kuzmin-tmi.scripts.inferred_triple_interaction_histogram_from_gene_selection]]
- [x] k=200 wetlab tables: 12 singles, 66 doubles C(12,2), 220 triples C(12,3) with overlay histogram [[select_12_k200_tables_hist|experiments.010-kuzmin-tmi.scripts.select_12_k200_tables_hist]]
- [x] Sameith overlap reference table [[select_12_experimental_table_reference|experiments.010-kuzmin-tmi.scripts.select_12_experimental_table_reference]]
- [x] gene list overlap analysis (UpSet) [[12_panel_gene_list_overlap|experiments.010-kuzmin-tmi.scripts.12_panel_gene_list_overlap]]
- [x] fixed Kemmeren sign inversion bug - cross-study correlation now +0.599 (was -0.599) [[verify_metadata|experiments.012-sameith-kemmeren.scripts.verify_metadata]]
- [x] single mutant expression distributions with improved stats (counts+percentages) [[single_mutant_expression_distributions|experiments.012-sameith-kemmeren.scripts.single_mutant_expression_distributions]]
- [x] combined triangular heatmap (Greens colormap for SD, RdBu for mean) [[double_mutant_combined_heatmap|experiments.012-sameith-kemmeren.scripts.double_mutant_combined_heatmap]]
- [x] gene-by-gene cross-study correlation analysis - confirms biological consistency [[gene_by_gene_expression_correlation|experiments.012-sameith-kemmeren.scripts.gene_by_gene_expression_correlation]]
- [x] Kemmeren-Sameith overlap analysis (mean-based, publication-quality plots) [[kemmeren_sameith_overlap_analysis|experiments.012-sameith-kemmeren.scripts.kemmeren_sameith_overlap_analysis]]
- [x] technical noise comparison using log2_ratio_std for both datasets [[noise_comparison_analysis|experiments.012-sameith-kemmeren.scripts.noise_comparison_analysis]]
- [x] master run_all script with stable filenames (no timestamps) [[run_all|experiments.012-sameith-kemmeren.scripts.run_all]]

## 2026.01.07

- [x] Construct refactor plan [[182525 Micro Rna Correcting Noise in Schema and for Datasets Kemmeren Sameit|scratch.2026.01.07.182525-micro-array-correcting-noise-in-schema-and-for-datasets-kemmeren-sameith]]
- [x] Describe some of the consquences of schema structure → [[2026.01.07 - Current Constraints on Schematization|torchcell.datamodels.schema#20260107---current-constraints-on-schematization]] → These manifest as sneaky design rules that could break later stuffy.

## 2026.01.08

- 🔲 On the [[2026.01.07 - Current Constraints on Schematization|torchcell.datamodels.schema#20260107---current-constraints-on-schematization]], do we specify the base class well enough to ensure that named properties are easily captured?

## 2026.01.09

- [x] prepare presentation
