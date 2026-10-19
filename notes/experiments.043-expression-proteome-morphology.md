---
id: r8tffo5bxgjy8v4x0qntim0
title: 043 Expression Proteome Morphology
desc: ''
updated: 1792393286898
created: 1792393286898
---

## 2026.10.19 - Design of stage 1: the raw store

Experiment 043 builds one store over every served KG 4.0 (release
`2026.10.17-4295f7ca`) dataset whose readout is a cell-state vector of expression,
protein abundance or turnover, or CalMorph morphology. Stage 1 is the query and the raw
store; nothing is trained.

### The 15 datasets

| dataset | phenotype class (recorded closure) | graph_level | gene clause | served experiments |
|---|---|---|---|---|
| MicroarrayKemmeren2014Dataset | MicroarrayExpressionPhenotype | node | all | 1,484 |
| SmMicroarraySameith2015Dataset | MicroarrayExpressionPhenotype | node | all | 82 |
| DmMicroarraySameith2015Dataset | MicroarrayExpressionPhenotype | node | all | 72 |
| MicroarrayScoreHu2007Dataset | MicroarrayExpressionScorePhenotype | node | all | 263 |
| TfDeletionPseudobulkJackson2020Dataset | SingleCellExpressionPhenotype | node | all | 784 |
| NadalRibellesPerturbSeq2025Dataset | PseudobulkExpressionPhenotype | node | all | 6,188 |
| ProteomeMessner2023Dataset | ProteinAbundancePhenotype | node | all | 4,699 |
| ProteomeZelezniak2018Dataset | ProteinAbundancePhenotype | node | all | 97 |
| ScmdOhya2005Dataset | CalMorphPhenotype | global | all | 4,718 |
| CaudalPanTranscriptome2024Dataset | RNASeqExpressionPhenotype | node | any | 943 |
| ProteomeMuenzner2024Dataset | ProteinAbundancePhenotype | node | any | 795 |
| ProteinTurnoverMuenzner2024Dataset | ProteinTurnoverPhenotype | node | any | 55 |
| Boocock2025Dataset | SingleCellExpressionPhenotype | node | none | 79,516 |
| Nguessan2025Dataset | SingleCellExpressionPhenotype | node | none | 2,951 |
| PooledPromoterExpressionVaishnav2022Dataset | PooledSortSeqExpressionPhenotype | node | none | 71,158 |

Served experiments are `n_experiments` in `database/releases/2026.10.17-4295f7ca.json`;
total 173,805. Gene clauses: `all` = every perturbation names a gene in `$gene_set` and
there is at least one; `any` = at least one does (the isolates carry accessory-ORF
absences and thousands of `sequence_variant` perturbations outside S288C); `none` = no
gene clause and an unlabeled genotype (`SegregantGenotype` for Boocock and Nguessan,
reached through `GenotypeMemberOf` because `segregant genotype` is_a `genotype` in
`biocypher/config/torchcell_schema_config.yaml`; a promoter-library reporter with no gene
for Vaishnav). Query and generator:
[[experiments.043-expression-proteome-morphology.scripts.generate_043_cql]]; registered as
the supported query `expression_proteome_morphology_043`.

### Five exclusions

- **Puddu 2019** (`genome_architecture_puddu2019`): a realized-genotype table, not a
  cell-state readout.
- **Ohnuki 2018**: shares 0 to 5 genes with any expression or proteome set.
- **Ohnuki 2022**: shares the target gene with other sets, not the strain.
- **Hackett 2020**: its induction perturbation would be read as a deletion by the
  `Perturbation` graph processor.
- **Messner 2021** (`proteome_messner2021`): no genotype perturbation.

### The raw-store rule

Owner's decision for the first query: keep the data. No converter, no deduplicator, no
genotype aggregation; one sample per served `(Experiment, ExperimentReference)` pair.
Reasons, from the reviewer's measurement and consistent with the source on main (v1.8.4):

- `MeanExperimentDeduplicator.duplicate_key` hashes `experiment_type` plus the sorted
  `systematic_gene_name` list only (`torchcell/data/mean_experiment_deduplicate.py`), so
  it ignores the environment and would average the same deletion across media (Jackson
  2020 measures 12 genotypes over 11 environment refs).
- `create_deduplicate_entry` raises `ValueError: Unsupported experiment type` for any type
  outside `_VECTOR_EXPERIMENT_TYPES` (`microarray_expression`, `rnaseq_expression`,
  `calmorph`, `metabolite`, `protein_abundance`, `visual_score`), which includes
  `pseudobulk_expression`, `pseudobulk_umi_count`, `microarray_expression_score` and
  `protein_turnover`.

How the rule is implemented (a deviation from `aggregator=None`, which does not run on
main): [[experiments.043-expression-proteome-morphology.scripts.build_043_store]].

### Shared label names

Three label names span more than one dataset, so a per-label readout sees several
sources under one name (dev census, 2026.10.19): `expression_log2_ratio` (Kemmeren,
Sameith Sm and Dm, Nadal-Ribelles), `umi_count` (Jackson, Boocock, Nguessan), and
`protein_abundance` (Messner 2023, Zelezniak, Muenzner proteome). The per-dataset label of every record is in
[[experiments.043-expression-proteome-morphology.scripts.store_census]].

### Scripts

- [[experiments.043-expression-proteome-morphology.scripts.generate_043_cql]]
- [[experiments.043-expression-proteome-morphology.scripts.build_043_store]]
- [[experiments.043-expression-proteome-morphology.scripts.gh_build_043_store]]
- [[experiments.043-expression-proteome-morphology.scripts.store_census]]

## 2026.10.19 - Stage 1 status

- Query generated, registered and validated (`validate` and `check` pass on
  `2026.10.17-4295f7ca`).
- Build job 5851 FAILED in the raw stage on the segregant genotypes (no `perturbations`
  list); nothing written. Details and traceback in
  [[experiments.043-expression-proteome-morphology.scripts.build_043_store]].
- Dev-tree census done (173,805 records); raw census waits on a successful build.

## 2026.10.19 - Stage 2: segregant fix, isolate clause, rebuild

- **torchcell fix (same PR):** a genotype with no `perturbations` list (a
  `SegregantGenotype`, by design) now counts as zero perturbed genes in
  `Neo4jQueryRaw.extract_systematic_gene_names`, `RawStageGrouping.prepare`, and the
  `Perturbation` and `DCellGraphProcessor` processors. The record is kept with empty
  `perturbation_indices` and `pert_mask` all False. Tests:
  `tests/torchcell/data/test_segregant_genotype_without_perturbations.py` (4 tests, 3 fail
  without the fix).
- **Isolate clause:** `(SIZE(perts) = 0 OR ANY(p IN perts WHERE p.systematic_gene_name IN
  $gene_set))`, so the Muenzner 2024 proteome strain with no perturbation is kept.
- **Known torchcell bug, not fixed here:** on main, `Neo4jCellDataset` with
  `aggregator=None` (and no deduplicator) copies the raw LMDB into `processed/`
  unchanged, keys `data_<i>` with one JSON object per value, while every reader expects
  keys `<i>` with a JSON list. Measured offline on a one-record LMDB: `get(0)` reads
  `None` and `compute_phenotype_info` raises `TypeError: string indices must be integers`.
  The resulting processed store cannot be opened by any reader. The 043 build works
  around it with the in-script `RecordAggregator` (one record per group).
- **Not fixed, same shape as the segregant bug:** the lazy index passes
  `compute_perturbation_count_index`, `compute_is_any_perturbed_gene_index` and
  `compute_is_any_deletion_gene_index` in `torchcell/data/neo4j_cell.py` still read
  `["genotype"]["perturbations"]` and catch exceptions per entry. Hypothesis
  (untested): they would skip segregant entries rather than raise. The 043 build does
  not run them: the folded raw-stage path writes those index files from the summaries.

## 2026.10.19 - Stage 3: two processor and index fixes; open items for the owner

Fixed in torchcell (same PR), with no re-query and no rebuild:

- **Shared label read once.** `MicroarrayExpressionPhenotype` and
  `PseudobulkExpressionPhenotype` both carry `expression_log2_ratio`. All six processors in
  `torchcell/data/graph_processor.py` that build `phenotype_types` and `stat_types` from
  `phenotype_info` now keep each name once, in order. Before the fix, the Kemmeren sample
  carried 12,338 values for 6,169 genes.
- **Segregants in the index passes.** `compute_perturbation_count_index`,
  `compute_is_any_perturbed_gene_index` and `compute_is_any_deletion_gene_index` read the
  perturbation list with `.get(...) or []`; a segregant counts 0 and names no gene. The
  index files were recomputed from the existing processed store
  ([[experiments.043-expression-proteome-morphology.scripts.reindex_043_store]]).

Build job 5856 requested 256 GB and `sstat` read a MaxRSS of 267 GB during the
aggregation and processed copies; the job was not killed.

### Open items for the owner

1. **188 records are dropped by the deletion blocks' `ALL` clause.** By dataset:
   Kemmeren 2, Hu 1, Nadal-Ribelles 70, Messner 75, Ohya 40. Each has a released gene name
   outside the 6,607-gene `SCerevisiaeGenome.gene_set`. Keeping them means the deletion
   blocks adopt the isolates' `SIZE = 0 OR ANY` shape, plus a re-query (about 6 h, from
   the 5 h 48 min raw stage of job 5855).
2. **Isolates enter the `Perturbation` processor with every `sequence_variant` gene marked
   perturbed.** The Caudal sample has 5,575 of 6,607 nodes in `pert_mask`. A rule over
   perturbation types is needed before any isolate training.
3. **`aggregator=None` on main writes an unreadable processed store.** The store keeps
   keys `data_<i>` with one object per value; `get(0)` reads `None` and
   `compute_phenotype_info` raises `TypeError`. The 043 build works around it with
   `RecordAggregator`; the bug is to be fixed separately.

## 2026.10.19 - Correlations between datasets of the same readout type

Script: [[experiments.043-expression-proteome-morphology.scripts.cross_dataset_correlation]]
(slurm job 5890 for extraction plus analysis, job 5892 for the final analysis and figures,
both from the raw LMDB and its cache; no re-query). Records are keyed by
(dataset, sorted perturbed systematic genes, environment hash). Isolates are keyed by
strain name, and Caudal and Muenzner name the same isolates.

### What each dataset reports, and what is compared

| dataset | label | released quantity | scale | used here |
|---|---|---|---|---|
| Kemmeren 2014 | expression_log2_ratio | two-channel array, deletion over wild type | log2 | as released |
| Sameith 2015 Sm | expression_log2_ratio | two-channel array, deletion over wild type | log2 | as released |
| Hu 2007 | expression_score | error-model X score vs BY4741 | signed score | as released |
| Nadal-Ribelles 2025 | expression_log2_ratio | pseudobulk log2 fold change vs wild type | log2 | as released; standard names mapped to ORFs |
| Jackson 2020 | umi_count | pseudobulk UMI counts (zeros omitted) | linear counts | log2((CPM+1)/(CPM_HO+1)) vs the HO deletion in the same environment; genes with at least 10 UMIs summed |
| Messner 2023 | protein_abundance | SWATH MaxLFQ batch-corrected | linear | log2(strain / record reference) |
| Zelezniak 2018 | protein_abundance | SWATH label-free log signal (SVA) | log | strain minus record reference |
| Muenzner 2024 proteome | protein_abundance | SWATH MaxLFQ precursor quantity | linear | log2, centered per protein over isolates |
| Muenzner 2024 turnover | degradation_rate | dynamic SILAC, per hour | linear | log2, centered per protein over isolates |
| Caudal 2024 | expression_tpm | RNA-seq TPM | linear | log2(TPM+1), centered per gene over isolates |
| Ohya 2005 | calmorph | CalMorph feature values | per feature | z per feature over shared genotypes, clipped at 8 |

Every pair compares log against log. Pearson and Spearman do not depend on the base of
the logarithm, so Zelezniak's unspecified log base does not change its pair.

### Headline: medians with IQR in brackets

| pair | n shared genotypes | n shared genes | per-genotype Pearson r, median [IQR] (n) | per-genotype Spearman rho, median [IQR] | per-gene Pearson r, median [IQR] (n genes) | fraction of genes with per-gene r > 0.3 |
|---|---|---|---|---|---|---|
| 1 Kemmeren x Sameith Sm | 82 | 6,169 | 1.000 [1.000, 1.000] (82) | 1.000 | 1.000 (6,169) | 1.000 |
| 2 Kemmeren x Hu | 255 | 5,803 | 0.107 [0.013, 0.184] (255) | 0.082 [0.000, 0.151] | 0.097 [0.023, 0.197] (5,803) | 0.102 |
| 3 Kemmeren x Nadal, YPD | 913 | 5,926 | 0.002 [-0.017, 0.022] (913) | 0.007 [-0.011, 0.027] | 0.017 [-0.010, 0.046] (5,891) | 0.007 |
| 3 Kemmeren x Nadal, YPD + 0.4 M NaCl 15 min | 920 | 5,926 | -0.001 [-0.021, 0.022] (920) | 0.005 [-0.014, 0.028] | 0.015 [-0.011, 0.043] (5,896) | 0.006 |
| 4 Kemmeren x Jackson YPD 10 h | 11 | 5,340 | 0.043 [0.026, 0.070] (11) | 0.047 [0.018, 0.058] | 0.035 [-0.217, 0.296] (5,340) | 0.247 |
| 4 Kemmeren x Jackson YPD 4 h (a) | 11 | 5,353 | 0.066 [0.027, 0.210] (11) | 0.030 [-0.006, 0.142] | 0.118 [-0.135, 0.381] (5,353) | 0.321 |
| 4 Kemmeren x Jackson YPD 4 h (b) | 11 | 5,601 | 0.076 [0.036, 0.149] (11) | 0.045 [0.026, 0.084] | 0.052 [-0.201, 0.304] (5,601) | 0.253 |
| 5a Nadal YPD x Nadal NaCl | 2,952 | 6,385 | 0.208 [0.168, 0.231] (2,952) | 0.043 [0.028, 0.059] | 0.183 [0.108, 0.223] (6,134) | 0.050 |
| 5b Nadal within-condition duplicates | 105 | 6,385 | 0.376 [0.323, 0.452] (105) | 0.178 [0.142, 0.240] | 0.062 [-0.020, 0.166] (5,897) | 0.089 |
| 6 Jackson, environment x environment per genotype | 605 genotype-environment pairs (11 genotypes, 11 environments) | per pair | 0.010 [-0.005, 0.029] (605) | 0.009 [-0.004, 0.030] | n/a | n/a |
| 7 Messner x Zelezniak | 89 | 715 | 0.049 [-0.016, 0.185] (89) | 0.050 [-0.030, 0.187] | 0.085 [-0.001, 0.176] (715) | 0.067 |
| 8 Kemmeren mRNA x Messner protein | 1,349 | 1,832 | 0.031 [-0.025, 0.112] (1,349) | 0.034 [-0.027, 0.113] | 0.094 [0.043, 0.157] (1,832) | 0.033 |
| 9a Caudal mRNA x Muenzner protein (isolates) | 759 | 1,572 | 0.208 [0.109, 0.307] (759) | 0.176 [0.085, 0.267] | 0.160 [0.084, 0.256] (1,572) | 0.183 |
| 9b Muenzner turnover x abundance (isolates) | 55 | 1,269 | -0.006 [-0.053, 0.047] (55) | -0.003 [-0.036, 0.049] | -0.007 [-0.126, 0.106] (1,120) | 0.066 |

Morphology link: ridge to CalMorph, 5-fold out-of-fold, statistic = Pearson r per
feature across held-out genotypes, then the mean and median over features. The earlier
number is the 019 `morphology_input_screen.json` (median over the 116 moving features).

| input matrix | n genotypes shared with Ohya | mean r over all 281 features | median r over all 281 | null mean r (all) | median r over the 116 moving features | earlier 019 median r over the 116 |
|---|---|---|---|---|---|---|
| Kemmeren 2014 | 1,437 | 0.238 | 0.238 | -0.005 | 0.348 | 0.345 (n=1,438) |
| Messner 2023 | 4,279 | 0.179 | 0.172 | 0.008 | 0.278 | 0.282 (n=4,314) |
| Hu 2007 | 259 | 0.114 | 0.105 | -0.052 | 0.235 | not run |
| Sameith 2015 Sm | 81 | -0.014 | -0.027 | -0.077 | -0.085 | not run |

The 116 moving-feature list is not on main. It was read from the retrospective
worktree's `results/calmorph_moving_features.json` (sha256 recorded in the json), so the
all-feature columns are the main-branch result.

Figures:

![](./assets/images/043-expression-proteome-morphology/cross_dataset_per_genotype_pearson_2026-10-19-14-37-41.svg)

![](./assets/images/043-expression-proteome-morphology/cross_dataset_mrna_protein_per_gene_2026-10-19-14-37-41.svg)

![](./assets/images/043-expression-proteome-morphology/cross_dataset_morphology_ridge_2026-10-19-14-37-41.svg)

### Readings (measured unless labeled)

- **Pair 1 is not a reproducibility ceiling.** All 82 Sameith 2015 single-mutant
  profiles are value-for-value identical to Kemmeren 2014's for the same deletions. This
  also holds on the dev LMDBs: 82 of 82 vectors match exactly. Hypothesis (untested):
  Sameith 2015 re-released the Kemmeren 2014 profiles of those strains; the Sameith SI
  should say. Until then the store holds 82 duplicated profiles under two dataset names,
  and Sameith Sm adds nothing to the morphology ridge (81 genotypes, mean r -0.014 against
  a null of -0.077).
- **Array expression agrees weakly with other platforms.**
  - Kemmeren against Hu 2007 (array, X score, different growth) gives a median
    per-genotype r of 0.107 over 255 TF deletions.
  - Against Jackson 2020 single-cell pseudobulk in YPD it gives 0.043 to 0.076 over 11
    TF deletions.
  - Against Nadal-Ribelles it gives 0.002 and -0.001 over about 915 deletions per
    condition.
- **Per-gene fractions for Jackson are at chance.** The Jackson per-gene fractions above
  0.3 (0.25 to 0.32) rest on 11 genotypes. The null spread of r at n = 11 is wide (about
  0.3), so these fractions are not distinguishable from chance. That is an expectation
  from sampling theory; this run computed no permutation null.
- **Nadal-Ribelles has a low internal ceiling.**
  - Duplicate records of the same genotype and condition agree at a median Pearson of
    0.376 (Spearman 0.178, n = 105 cells).
  - The same genotype across YPD and NaCl agrees at Pearson 0.208 (Spearman 0.043,
    n = 2,952).
  - Pearson is well above Spearman, so the agreement sits in a few large values. The
    log2 fold changes reach plus or minus 28. Hypothesis (untested): these come from genes
    with near-zero counts on one side.
  - With within-dataset duplicates at 0.38, an agreement of about 0 with Kemmeren is
    within what the Nadal measurement can resolve, not evidence against a shared signal.
- **Jackson TF-deletion profiles do not repeat across environments.**
  - The median r of the HO-ratio profiles across 55 environment pairs per genotype is
    0.010 (605 pairs).
  - The matrix itself is sound: the log2 CPM level profiles of two genotypes in one
    environment correlate at 0.959.
  - The r rises with the UMI floor: 0.052 at 200 UMIs and 0.097 at 1,000 (a diagnostic run
    on the cache, not in the json).
  - Hypothesis (untested): count noise in pseudobulk of about 6 records per cell
    dominates the ratio.
- **Protein against protein (Messner x Zelezniak), same lab:** 0.049 median per-genotype
  r over 89 kinase deletions and 715 proteins.
- **mRNA against protein.**
  - Deletion strains (Kemmeren x Messner) correlate at 0.031 per genotype (1,349
    deletions) and 0.094 per gene (1,832 genes).
  - Natural isolates (Caudal x Muenzner) correlate at 0.208 per isolate (759) and 0.160
    per gene (1,572).
  - The isolates vary far more across strains, as the protein SD axis of the scatter
    figure shows. Hypothesis (untested): that larger range is why mRNA and protein agree
    more there.
  - Turnover against abundance is about 0 (55 isolates).
  - The 3,317 expression-by-protein genotypes of the co-location census include
    Nadal-Ribelles; Kemmeren alone shares 1,349 with Messner.
- **Morphology.** Kemmeren carries the most CalMorph signal per genotype: mean r 0.238 over
  all 281 features, against Messner 0.179, Hu 0.114 and Sameith Sm -0.014. On the
  116 moving features, Kemmeren (0.348) and Messner (0.278) reproduce the 019 screen
  (0.345 and 0.282) on the raw store.
- **Boocock against N'Guessan: no comparison is possible.**
  - Shared segregant ids: 0 of 79,516 and 2,951.
  - Shared crosses: none. Boocock has crosses `3004`, `A` and `B` (BY with RM, CBS2888,
    YJM145, YJM981 and YPS163); N'Guessan has `BBQTL_BYxRM` (BY4741 x RM11-1a).
  - Both name RM as a parent, but under different strain names and as different
    segregant panels.
