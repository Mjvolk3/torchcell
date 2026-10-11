---
id: rdi64el69j2nxjlvtrcftqu
title: Kuzmin2020
desc: ''
updated: 1725990105586
created: 1723828982525
---

## LLM Prompt for Initial Build of Dataset Before Debug

We are going to use raw data from `kuzmin2020` to create the following dataset:

- `SmfKuzmin2020Dataset`
- `DmfKuzmin2020Dataset`
- `TmfKuzmin2020Dataset`
- `DmiKuzmin2020Dataset`
- `TmiKuzmin2020Dataset`

Here is the new link address. "<https://www.science.org/doi/suppl/10.1126/science.aaz5667/suppl_file/aaz5667-tables_s1_to_s13.zip>"

from within this zip we need aaz5667-Table-S1... it has 794774 total rows

Here is a snippets of the `xlsx` file with headers.

Table S1. Raw genetic interaction dataset. This file contains digenic interaction scores as well as raw and adjusted trigenic interaction scores with 12 columns.

```xlsx
Query strain ID Query allele name Array strain ID Array allele name Combined mutant type Raw genetic interaction score (epsilon) Adjusted genetic interaction score (epsilon or tau) P-value Query single/double mutant fitness Array single mutant fitness Double/triple mutant fitness Double/triple mutant fitness standard deviation
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL007C_dma91 sla1Δ digenic 0.007534 0.007534 4.50E-01  0.962 0.9695 0.0465
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL008W_dma90 hir1Δ digenic -0.008696 -0.008696 3.30E-01  1.02 1.0113 0.0123
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL021C_dma117 hap3Δ digenic -0.096643 -0.096643 1.68E-01  0.852 0.7554 0.1147
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL023C_tsa111 mcm2-1 digenic 0.005416 0.005416 4.65E-01  0.909 0.9144 0.0479
YAL015C+YDL227C_tm461 ntg1Δ+hoΔ YBL024W_dma116 ncl1Δ digenic -0.047343 -0.047343 1.81E-01  0.966 0.9187 0.0444
```

Notice that just like kuzmin 2018 we have to deal with YDL227C which is a hoΔ. This ids part of total number of gene interactions.

We can combine Table-S1 with S3 directly

Table S3. Raw genetic interaction dataset from pilot screens. This file contains digenic interaction scores as well as raw and adjusted trigenic interaction scores in a tab-delimited format with 12 columns.

This is file name aaz5667-Table-S3.xlsx

```
Query strain ID Query allele name Array strain ID Array allele name Combined mutant type Raw genetic interaction score (epsilon) Adjusted genetic interaction score (epsilon or tau) P-value Query single/double mutant fitness Array single mutant fitness Double/triple mutant fitness Double/triple mutant fitness standard deviation
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL002W_dma23 vps8Δ digenic 0.052845 0.052845 1.98E-01  0.764 0.8168 0.0501
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL004W_dma22 yal004wΔ digenic -0.00189 -0.00189 4.90E-01  1.009 1.0071 0.075
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL005C_dma21 ssa1Δ digenic 0.006864 0.006864 4.61E-01  1.0405 1.0474 0.0761
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL007C_dma20 erp2Δ digenic -0.020312 -0.020312 3.80E-01  1.026 1.0057 0.0699
YAL056W+YDL227C_tm1888 gpb2Δ+hoΔ YAL008W_dma19 fun14Δ digenic 0.002584 0.002584 4.82E-01  1.0445 1.0471 0.0497
...
```

We also need to use Table S5 aaz5667-Table-S5.xlsx. Their is either 'Double mutant' or 'Single mutant' here

```xlsx
"Table S5. Fitness standard for single and double mutant query strains and query-query interactions.
This file contains the fitness standard for single and double mutant query strains as well as the genetic interaction between the query genes. "         
Query Strain ID ORF1 ORF2 Gene1 Gene2 Allele1 Allele2 Mutant type Fitness St.dev.
tm1 YAR002C-A YGL002W ERP1 ERP6 erp1Δ erp6Δ Double mutant 0.87051 0.007
tm10 YDR492W YOL101C IZH1 IZH4 izh1Δ izh4Δ Double mutant 0.89413 0.005
tm100 YIR037W YKL026C HYR1 GPX1 hyr1Δ gpx1Δ Double mutant 0.99379 0.0045
tm101 YDR146C YLR131C SWI5 ACE2 swi5Δ ace2Δ Double mutant NaN NaN
tm103 YIL117C YNL058C PRM5 YNL058C prm5Δ ynl058cΔ Double mutant 0.92621 0.0055
tm104 YMR182C YPL230W RGM1 USV1 rgm1Δ usv1Δ Double mutant 0.88059 0.0045
...
```

To summarize the files that will be necessary for each dataset. These should also be the files that are used to create the raw dataset.

- `SmfKuzmin2020Dataset`
  files needed:
  - aaz5667-Table-S5.xlsx
- `DmfKuzmin2020Dataset`
  - aaz5667-Table-S1.xlsx
  - aaz5667-Table-S3.xlsx
  - aaz5667-Table-S5.xlsx
- `TmfKuzmin2020Dataset`
  - aaz5667-Table-S1.xlsx
- `DmiKuzmin2020Dataset`
  - aaz5667-Table-S1.xlsx
- `TmiKuzmin2020Dataset`
  - aaz5667-Table-S1.xlsx

To implement this make sure we follow the patterns laid out by `kuzmin2018` datasets, and adhere to the schema.py.

This is the DOI: 10.1126/science.aaz5667, and pubmed id 32586993 and pubmed url <https://pubmed.ncbi.nlm.nih.gov/32586993/>

## 2024.09.10 - There are Negative Fitness Values

I didn't think we would ever encounter this since the fitness score is computed as the ratio of mutant over wild type. This is likely an artifact of data processing, normalization, standardization, etc. Based on this definition it is strange to keep the negative as it violates our current data validation. [[torchcell.datamodels.schema]]. I think for this instance we should correct to 0.0 with the validator but provide a warning. I think this is okay given so few data points are negative and for all practical purpose negative means `0.0` or a dead cell.

Table S3.

| Query strain ID        | Query allele name | Array strain ID | Array allele name | Combined mutant type | Raw genetic interaction score (epsilon) | Adjusted genetic interaction score (epsilon or tau) | P-value  | Query single/double mutant fitness | Array single mutant fitness | Double/triple mutant fitness | Double/triple mutant fitness standard deviation |
| ---------------------- | ----------------- | --------------- | ----------------- | -------------------- | --------------------------------------- | --------------------------------------------------- | -------- | ---------------------------------- | --------------------------- | ---------------------------- | ----------------------------------------------- |
| YMR198W+YDL227C_tm467  | cik1Δ+hoΔ         | YPR162C_tsa1247 | orc4-5001         | digenic              | -0.718108                               | -0.718108                                           | 1.71E-57 |                                    | 0.6379                      | -0.0802                      | 0.0671                                          |
| YAL056W+YOR371C_tm1601 | gpb2Δ+gpb1Δ       | YNL102W_tsa345  | pol1-17           | trigenic             | -0.523199                               | 0.160684                                            | 0.00E+00 | 0.9083                             | 0.5481                      | -0.0254                      | 0                                               |
| YKR095W+YIL149C_tm1623 | mlp1Δ+mlp2Δ       | YMR197C_tsa635  | vti1-11-supp1     | trigenic             | -0.630539                               | -0.818542                                           | 4.35E-35 | 0.8307                             | 0.7494                      | -0.008                       | 0.0559                                          |

Table S1.

| Query strain ID        | Query allele name | Array strain ID | Array allele name | Combined mutant type | Raw genetic interaction score (epsilon) | Adjusted genetic interaction score (epsilon or tau) | P-value   | Query single/double mutant fitness | Array single mutant fitness | Double/triple mutant fitness | Double/triple mutant fitness standard deviation |
| ---------------------- | ----------------- | --------------- | ----------------- | -------------------- | --------------------------------------- | --------------------------------------------------- | --------- | ---------------------------------- | --------------------------- | ---------------------------- | ----------------------------------------------- |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YBR131W_dma294  | ccz1Δ             | digenic              | -0.872019                               | -0.872019                                           | 0.00E+00  |                                    | 0.744                       | -0.128                       | 0                                               |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YPL226W_dma4968 | new1Δ             | digenic              | -0.375281                               | -0.375281                                           | 1.11E-85  |                                    | 0.327                       | -0.0483                      | 0.0149                                          |
| YHR030C+YKL161C_tm61   | slt2Δ+kdx1Δ       | YGR157W_dma1899 | cho2Δ             | trigenic             | -0.256726                               | -0.045604                                           | 8.74E-38  | 0.8665                             | 0.247                       | -0.0427                      | 0.0196                                          |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YFL007W_dma5286 | blm10Δ            | digenic              | -1.077915                               | -1.077915                                           | 0.00E+00  |                                    | 1.036                       | -0.0419                      | 0.0275                                          |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YHR003C_dma2083 | tcd1Δ             | digenic              | -1.006909                               | -1.006909                                           | 0.00E+00  |                                    | 0.965                       | -0.0419                      | 0                                               |
| YCR069W+YNR028W_tm163  | cpr4Δ+cpr8Δ       | YGL023C_dma1597 | pib2Δ             | trigenic             | -0.876595                               | -0.850199                                           | 2.52E-61  | 0.8946                             | 0.943                       | -0.033                       | 0.056                                           |
| YHR030C+YDL227C_tm432  | slt2Δ+hoΔ         | YGR157W_dma1899 | cho2Δ             | digenic              | -0.280035                               | -0.280035                                           | 1.89E-12  |                                    | 0.247                       | -0.033                       | 0.0527                                          |
| YDR077W+YER150W_tm3410 | sed1Δ+spi1Δ       | YAL025C_tsa1066 | MAK16-ph          | trigenic             | -0.741312                               | -0.722245                                           | 0.00E+00  |                                    | 0.716                       | -0.0253                      | 0                                               |
| YER059W+YDL227C_tm545  | pcl6Δ+hoΔ         | YKR092C_dma3002 | srp40Δ            | digenic              | -0.952263                               | -0.952263                                           | 0.00E+00  |                                    | 0.927                       | -0.0253                      | 0                                               |
| YER059W+YDL227C_tm545  | pcl6Δ+hoΔ         | YLR077W_dma3164 | fmp25Δ            | digenic              | -1.012256                               | -1.012256                                           | 0.00E+00  |                                    | 0.996                       | -0.0163                      | 0.0119                                          |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YLR077W_dma3164 | fmp25Δ            | digenic              | -1.011758                               | -1.011758                                           | 4.23E-120 |                                    | 0.996                       | -0.0158                      | 0.048                                           |
| YER059W+YDL227C_tm545  | pcl6Δ+hoΔ         | YPL226W_dma4968 | new1Δ             | digenic              | -0.342596                               | -0.342596                                           | 0.00E+00  |                                    | 0.327                       | -0.0156                      | 0                                               |
| YMR199W+YPL256C_tm1604 | cln1Δ+cln2Δ       | YLR337C_dma3380 | vrp1Δ             | trigenic             | -0.355614                               | -0.355614                                           | 5.81E-03  |                                    | 0.53                        | -0.0156                      | 0.0901                                          |
| YDR483W+YDL227C_tm704  | kre2Δ+hoΔ         | YGR121C_dma1881 | mep1Δ             | digenic              | -1.024336                               | -1.024336                                           | 0.00E+00  |                                    | 1.01                        | -0.0143                      | 0.0209                                          |
| YAL056W+YDL227C_tm1888 | gpb2Δ+hoΔ         | YHR081W_dma2126 | lrp1Δ             | digenic              | -0.499073                               | -0.499073                                           | 3.99E-09  |                                    | 0.49                        | -0.0091                      | 0.2276                                          |

## 2026.09.20 - Ingest the double-mutant query strain fitness

The published trigenic score needs the double-mutant query strain's own fitness, and the loader's attempt to bring it in from Table S5 never matched a row: it joined the full strain string against the bare tm number, matched 0 of 632,797 rows, and the `fillna` fallback silently used the S1/S3 columns, so the mismatch warning could never fire. The row set was also wrong, because an S5 double-mutant entry describes a trigenic row's query strain, not a digenic pair.

- One record per distinct double-mutant query strain, 201 of them, with the query pair as the genotype and the query strain id on both perturbations (main `4c4a4f950`).
- The released S5 standard deviation is stored as a bootstrap SE over the conservative lower end of the 12 to 24 colony measurements the SI describes.
- Digenic records serialize byte-identically, which is what lets the served graph take the additions as a superset admission [[torchcell.knowledge_graphs.incremental-admission]].

## 2026.10.01 - Issue #533 Refusals and an Entry Point Under DATA_ROOT

- An array strain that is neither `tsa` nor `dma` used to be stored as an `SgaAllelePerturbation` by Dmf, Dmi, Tmf and Tmi, a guess with no evidence. `_array_perturbation_type` now refuses it with the same named `ValueError` as the 2018 loaders (0 of 934,595 Table S1 and S3 rows).
- A blank SD used to be stored as `fitness_std` NaN on Smf, Dmf (both record kinds) and Tmf. `_reported_sd` stores None, matching the empty uncertainty fields; the SI defines no meaning for a blank (`kuzminExploringWholegenomeDuplicate2020/si/si1.md` lists only "12. Combined mutant fitness standard deviation", line 229). Measured on the real tables: 0 blank St.dev. among the 472 Smf rows and 201 query-strain records, 0 blank SDs in S1 and S3.
- A Table S5 "Double mutant" tm number listed twice fanned the left merge out into two records of one strain. `_double_mutant_query_strain_rows` now refuses it (0 of 240 released rows repeat).
- `main` built only `TmiKuzmin2020Dataset()` at the relative default root; it now builds all five under `$DATA_ROOT/data/torchcell/<loader>_kuzmin2020` and refuses an unset `DATA_ROOT` through `torchcell.graph.sgd.data_root`.
- Observation, not changed (filed as its own dataset issue): Table S1 is the main screen ("Table S1. Raw genetic interaction dataset.", `kuzminExploringWholegenomeDuplicate2020/si/si1.md` line 200) and Table S3 the pilot screens ("Table S3. Raw genetic interaction dataset from pilot screens.", line 257; screening method at lines 57 and 69). The loaders concatenate both, so a (query strain, array strain) key measured in both tables is stored twice. Measured on the pinned tables (`verify_s1_s3_overlap.py`): digenic, 24,193 keys in both tables, 24,179 differing in fitness (mean |diff| 0.066, max 0.661), 1,460 with a different array allele name; trigenic, 11,443 keys in both, 11,436 differing in fitness (mean |diff| 0.061, max 0.792), 686 with a different array allele name. The trigenic overlap feeds Tmf and Tmi 2020. No key repeats within either table.
- Review follow-up tests: `test_tmf_blank_sd_is_stored_as_none`, `test_array_strain_type_is_matched_case_sensitively`, and an S5 repeat with a different fitness (0.51 against 0.5).

## 2026.10.02 - Issue #602 Main and Pilot Screens Carry Their Screen

- **Problem.** Table S1 (main screen against the ~1,200-strain diagnostic array) and Table S3 (pilot screens against the genome-wide NES and ES arrays, `kuzminExploringWholegenomeDuplicate2020/si/si1.md` lines 21, 57, 69, 200, 257) both release 24,193 digenic and 11,443 trigenic (query strain, array strain) keys. 24,179 and 11,436 of them differ in fitness, and 0 have the same value in both tables (fitness and SD for Dmf/Tmf, score and p for Dmi/Tmi). Both rows were stored with nothing naming the screen, so one (genotype, environment) held two values that could not be told apart. The two tables also spell 70 of the 1,159 shared array strains differently (19 by case only, `tfc3-G349E` / `tfc3-g349e`; 50 `-ph` / `-PH` placeholders against a numbered allele, `PRE7-ph` / `pre7-5001`; 1 renamed ORF, `ymr166cΔ` / `mme1Δ`), which gave one strain two `perturbed_gene_name` values on 1,460 digenic and 686 trigenic keys.
- **Decision (orchestrator's choice, open to reversal by the owner): keep both measurements.** Each one is a replicate from a different screen. The phenotype now records which screen it came from.
- **Schema.** `FitnessPhenotype.screen_id` and `GeneInteractionPhenotype.screen_id` (`str | None`, default None) carry the same meaning as `EnvironmentResponsePhenotype.screen_id`. The cell adapter projects `screen_id` onto the fitness and gene-interaction phenotype nodes (experiment and reference), and `biocypher/config/torchcell_schema_config.yaml` declares it. Every dataset that stores these phenotypes now serializes `"screen_id": null` (Costanzo 2016, Kuzmin 2018, Baryshnikova 2010, O'Duibhir 2014), so this is a full-rebuild change.
- **Loader.** `_combine_screens` concatenates S1 then S3 and tags each row: `kuzmin2020_s1_diagnostic_array`, `kuzmin2020_s3_pilot_genome_wide_arrays`. Smf records and Dmf query-strain records read their value from Table S5, so they carry `kuzmin2020_s5_query_fitness_array` (SI lines 39 to 41). A query-strain fallback to the S1/S3 column would keep the S1/S3 screen; 0 of the 201 stored strains fall back.
- **One array strain, one name.** The strain id is identical in both tables, so a spelling difference is one physical strain, not two alleles. Every row of a strain that Table S1 lists uses S1's `Array allele name`; S3's name is kept only for strains S1 does not list. A strain named two ways inside ONE table is refused (0 such strains in either released table). Each of the 70 strain differences is a case, placeholder or ORF-rename difference; none names a different mutation.
- **Measured** (`experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2020_screen_source.py`, results `experiments/036-dataset-fixes-before-kg-build/results/kuzmin2020_screen_source.json`, preprocess stage on the released tables, no LMDB built):
  - Rows per screen: Dmf 537,911 main + 94,886 pilot + 201 query-strain; Dmi the same 537,911 + 94,886; Tmf and Tmi 256,861 + 44,937; Smf 472, all S5. The Dmf, Dmi, Tmf and Tmi row counts (632,998 / 632,797 / 301,798 / 301,798) equal the served counts the issue quotes for release `2026.09.21-ab6d8c5d`, so no record is added or dropped.
  - S3 rows renamed to the S1 spelling: 2,147, on 70 strains. Array strains with more than one name after preprocessing: 0 of 4,553.
  - Experiment ids (sha256 of the experiment dump, as `cell_adapter._experiment_node` computes it): every record is distinct, with or without `screen_id`, because no repeated key has identical values. (Genotype, environment) groups holding more than one record: Dmf and Dmi 24,193, Tmf and Tmi 11,443. Every group holds exactly 2 records, one from each screen. Before the renaming, the 1,460 and 686 keys with two spellings were not in one group at all, because they had two genotypes.
- **Deduplicator.** `MeanExperimentDeduplicator` (used by 006 and 010) keys on experiment type + sorted systematic gene names, not on the phenotype. A trigenic training query therefore still sees each main/pilot pair as ONE group and mean-pools the two values; `screen_id` does not split the group. The pooled record is built without a `screen_id` (None), and its `dataset_name` is the `+`-joined source names. Treating the pair as two separate replicates would need a change to the deduplicator's key. Not changed here.
- **Tests.** `test_a_cross_in_both_screens_is_two_records_named_by_screen` and `test_a_triple_in_both_screens_keeps_both_with_the_s1_array_name` are hermetic. Data-gated, real tables: `test_issue_602_pair_is_two_records_one_per_screen` (gpb2 x tfc3-G349E: Dmf 0.9558 / 0.0188 main and 0.8746 / 0.0171 pilot; Dmi -0.013157 and 0.008843) and `test_released_digenic_rows_per_screen_and_one_name_per_array`. `test_two_names_for_one_array_strain_within_one_table_are_refused` pins the refusal.
- **Open.**
  - The five dev stores need a slurm rebuild.
  - The supported-query gate reports `contract_changed` drift for `essentiality_smf` and `solid_growth_025` (FitnessPhenotype, GeneInteractionPhenotype). After the next KG build they need re-validation against the new release.
  - Whether the deduplicator should pool the two screens or keep them apart is still to decide.

## 2026.10.10 - L0-L4 verification of the dev stores (#889)

Fitness stores run the existing fitness battery unchanged through its single-pass twin `verify_fitness_dataset_streaming` (L0 validates each record as `FitnessExperiment`; every rule of the eager verifier applies, none skipped). `pair_uniqueness` keys on the record's perturbation SET plus strain ids and `screen_id`, so the double and triple arms needed no host-aware variant (the #889 hypothesis held: 0 duplicates on every Kuzmin arm). Interaction stores run the new gene-interaction battery (`torchcell/verification/gene_interaction.py`): every rule applies; `released_values` compares the stored (score, p-value) multiset to the sha256-pinned release, and `fitness_companion_containment` checks every (strain, environment) key against the companion fitness store.

Release counts equal the stores: 632,797 digenic rows (537,911 Table S1 + 94,886 Table S3); 301,798 trigenic rows; DMF adds the 201 of 240 trigenic double-mutant query strains whose fitness Table S5 states (632,998); SMF is the 472 of 480 Table S5 single-mutant rows that state a fitness.

Findings, each left failing and filed: 47 released ORFs are retired or renamed in R64-4-1 (`canonical_gene_names`, `current_genome_genes`) in all four multi-mutant arms, #897 (`dmf_kuzmin2020`, also 33 released SDs of exactly 0), #898 (`tmf_kuzmin2020`), #899 (`dmi_kuzmin2020`), #900 (`tmi_kuzmin2020`). `smf_kuzmin2020` passes every rule. Observation, no rule: 18 released double/triple fitness values are negative in Tables S1/S3 and `FitnessPhenotype` clamps them to 0 (the 2024.09.10 section above), so the stored fitness differs from the release on those rows.

Tables generated by `experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables.py` from the reports the sweep wrote (`kg4_pre_build_verification_sweep.py --dataset <Class>`, read-only on `processed/`); count oracles derived by `no_verifier_store_release_counts.py` (results `results/no_verifier_store_release_counts.json`).

`smf_kuzmin2020`: **PASS**, 14 rules, 0 failing (report written 2026-10-10T21:04:30)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 472 records validated |
| L1 | `count` | PASS | observed 472, expected 472 |
| L1 | `pair_uniqueness` | PASS | 472 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 472 values checked |
| L2 | `se_nonnegative` | PASS | 472 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 472 records |
| L1 | `provenance_gaps` | PASS | 472 documented provenance gaps over 472/472 records; 0 deferred field(s): []; 15104 undeclared None values over 8 carrier fields (top: Compound.inchi x6136, Compound.chebi_id x4248, Compound.smiles x1416, Compound.inc... |
| L1 | `canonical_gene_names` | PASS | 235 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | PASS | 472 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 3304 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 472 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 235 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 235 measured systematic names is a gene of the current genome |

`dmf_kuzmin2020`: **FAIL**, 14 rules, 3 failing (report written 2026-10-10T21:10:57)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 632998 records validated |
| L1 | `count` | PASS | observed 632998, expected 632998 |
| L1 | `pair_uniqueness` | PASS | 632998 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 632998 values checked |
| L2 | `se_nonnegative` | PASS | 632998 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 632998 records |
| L1 | `provenance_gaps` | PASS | 632998 documented provenance gaps over 632998/632998 records; 0 deferred field(s): []; 20255936 undeclared None values over 8 carrier fields (top: Compound.inchi x8228974, Compound.chebi_id x5696982, Compound.smiles x... |
| L1 | `canonical_gene_names` | **FAIL** | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 47 systematic names are not the genome's current name; 40 common names resolve to another gene |
| L2 | `uncertainty_sanity` | **FAIL** | 33/632998 labeled uncertainties are a sample dispersion of exactly 0 (a pseudocount artifact, not perfect precision) |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 4430986 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 632998 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 0.989 of 4367 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | **FAIL** | 47 systematic names are absent from the current genome over 1000 records: YAL043C-A x21, YAL058C-A x21, YAR037W x22, YAR040C x22, YAR043C x22, YAR044W x22, YBR075W x21, YBR090C-A x21, YBR100W x21, YCL006C x21 |

`tmf_kuzmin2020`: **FAIL**, 14 rules, 2 failing (report written 2026-10-10T21:14:50)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 301798 records validated |
| L1 | `count` | PASS | observed 301798, expected 301798 |
| L1 | `pair_uniqueness` | PASS | 301798 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 301798 values checked |
| L2 | `se_nonnegative` | PASS | 0 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 301798 records |
| L1 | `provenance_gaps` | PASS | 301798 documented provenance gaps over 301798/301798 records; 0 deferred field(s): []; 11166526 undeclared None values over 13 carrier fields (top: Compound.inchi x3923374, Compound.chebi_id x2716182, Compound.smiles ... |
| L1 | `canonical_gene_names` | **FAIL** | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 47 systematic names are not the genome's current name; 40 common names resolve to another gene |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 2112586 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 301798 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 0.989 of 4367 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | **FAIL** | 47 systematic names are absent from the current genome over 483 records: YAL043C-A x10, YAL058C-A x10, YAR037W x11, YAR040C x11, YAR043C x11, YAR044W x11, YBR075W x10, YBR090C-A x10, YBR100W x10, YCL006C x10 |

`dmi_kuzmin2020`: **FAIL**, 23 rules, 2 failing (report written 2026-10-10T21:23:15)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 632797 records validated as GeneInteractionExperiment |
| L0 | `reference_structural` | PASS | 632797 records validated as GeneInteractionExperimentReference |
| L1 | `count` | PASS | observed 632797, expected 632797 |
| L1 | `pair_uniqueness` | PASS | 632797 unique (strain, environment) records, one each |
| L1 | `interaction_order` | PASS | all 632797 records perturb 2 distinct genes at graph level 'edge' |
| L2 | `score_finite` | PASS | 632797 interaction scores are finite numbers |
| L2 | `p_value_in_unit_interval` | PASS | 632797 stated p-values lie in [0, 1] (0 records state none) |
| L2 | `released_values` | PASS | the 632797 stored (score, p-value) pairs are the released multiset (the 'digenic' rows of Tables S1 and S3) |
| L3 | `reference_zero` | PASS | every one of the 632797 reference scores is 0 with no p-value, at the experiment's graph level |
| L3 | `reference_environment` | PASS | every one of the 632797 references is measured in its experiment's own environment |
| L3 | `signed_unclamped` | PASS | 336374 aggravating and 296416 alleviating scores, 7 at exactly zero |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L1 | `provenance_gaps` | PASS | 632797 documented provenance gaps over 632797/632797 records; 0 deferred field(s): []; 22780692 undeclared None values over 12 carrier fields (top: Compound.inchi x8226361, Compound.chebi_id x5695173, Compound.smiles ... |
| L1 | `canonical_gene_names` | **FAIL** | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 47 systematic names are not the genome's current name; 40 common names resolve to another gene |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 4429579 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 632797 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 0.989 of 4367 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | **FAIL** | 47 systematic names are absent from the current genome over 1000 records: YAL043C-A x21, YAL058C-A x21, YAR037W x22, YAR040C x22, YAR043C x22, YAR044W x22, YBR075W x21, YBR090C-A x21, YBR100W x21, YCL006C x21 |
| L4 | `fitness_companion_containment` | PASS | every one of the 632797 (strain, environment) keys is a record of dmf_kuzmin2020 |

`tmi_kuzmin2020`: **FAIL**, 23 rules, 2 failing (report written 2026-10-10T21:28:25)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 301798 records validated as GeneInteractionExperiment |
| L0 | `reference_structural` | PASS | 301798 records validated as GeneInteractionExperimentReference |
| L1 | `count` | PASS | observed 301798, expected 301798 |
| L1 | `pair_uniqueness` | PASS | 301798 unique (strain, environment) records, one each |
| L1 | `interaction_order` | PASS | all 301798 records perturb 3 distinct genes at graph level 'hyperedge' |
| L2 | `score_finite` | PASS | 301798 interaction scores are finite numbers |
| L2 | `p_value_in_unit_interval` | PASS | 301798 stated p-values lie in [0, 1] (0 records state none) |
| L2 | `released_values` | PASS | the 301798 stored (score, p-value) pairs are the released multiset (the 'trigenic' rows of Tables S1 and S3) |
| L3 | `reference_zero` | PASS | every one of the 301798 reference scores is 0 with no p-value, at the experiment's graph level |
| L3 | `reference_environment` | PASS | every one of the 301798 references is measured in its experiment's own environment |
| L3 | `signed_unclamped` | PASS | 140885 aggravating and 160908 alleviating scores, 5 at exactly zero |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L1 | `provenance_gaps` | PASS | 301798 documented provenance gaps over 301798/301798 records; 0 deferred field(s): []; 10864728 undeclared None values over 12 carrier fields (top: Compound.inchi x3923374, Compound.chebi_id x2716182, Compound.smiles ... |
| L1 | `canonical_gene_names` | **FAIL** | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 47 systematic names are not the genome's current name; 40 common names resolve to another gene |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 2112586 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 301798 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 0.989 of 4367 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | **FAIL** | 47 systematic names are absent from the current genome over 483 records: YAL043C-A x10, YAL058C-A x10, YAR037W x11, YAR040C x11, YAR043C x11, YAR044W x11, YBR075W x10, YBR090C-A x10, YBR100W x10, YCL006C x10 |
| L4 | `fitness_companion_containment` | PASS | every one of the 301798 (strain, environment) keys is a record of tmf_kuzmin2020 |

## 2026.10.11 - Released ORFs that R64-4-1 no longer carries: 16 renamed, 31 refused (#897, #898, #899, #900)

- **Finding.** The #889 verifier (PR #980) failed `current_genome_genes` and `canonical_gene_names` on Dmf, Tmf, Dmi and Tmi: 47 systematic names absent from R64-4-1 (1,000 Dmf/Dmi and 483 Tmf/Tmi records), and 40 common names resolving to another gene. Smf passed.
- **Classification** (measured by [[experiments.036-dataset-fixes-before-kg-build.scripts.yeast_gene_id_currency]], slurm job 3829): 16 renamed, 25 merged into a longer locus (23 genes, 2 blocked reading frames), 6 retired. All 47 are array strains of the Table S3 pilot screens against the genome-wide arrays. The 16 renames are SGD nomenclature corrections of the same locus (for example "YAL043C-A used to be the systematic name for this ORF. Since this name was incorrect, this ORF name has been changed to YAL042C-A to properly represent the systematic ORF nomenclature.", SGD locus history of YAL042C-A); for 13 of them Costanzo 2016 names the same strain number by the current ORF (`YAL043C-A_dma42` is `YAL042C-A_dma42` there). Ten of the 16 renamed loci are Dubious ORFs that R64-4-1 still carries. Every merged ORF sits beside a separate deletion strain of its locus in Tables S1/S3. YFL035C-B is renamed to YFL034C-A, which also has its own strain (`YFL034C-A_dma1493` beside `YFL035C-B_dma1491`); both are stored as YFL034C-A, told apart by strain id.
- **Rule and loader change** as in [[torchcell.datasets.systematic_name_currency]]: all five loaders take `genome` and check the systematic names each record stores (Dmf per record kind: the digenic query and array, then the query-strain pair), after the reference noise, and write `preprocess/systematic_name_ledger.json`. A renamed ORF keeps its released strain id and allele name (`YAL043C-A_dma42`, `yal043c-a`) and stores the current systematic name.
- **imp2'.** The 40th common-name mismatch was the loader's, not the release's: `imp2'Δ` (YIL154C) went through `'` -> `_prime` and `split("_")[0]`, so it was stored as `imp2`, the standard name of YMR035W. `_stored_gene_name` restores the prime, so the strain is `imp2'` as Costanzo 2016 stores it (R64-4-1 lists IMP2' as an alias of YIL154C alone).
- **Records per store** (input frame measured by job 3829; confirmed on the rebuilt stores below):

| Store | Records before | Refused (merged + retired) | Records after | Renamed (systematic name changed) | imp2' (common name changed) |
|---|---:|---:|---:|---:|---:|
| smf_kuzmin2020 | 472 | 0 | 472 | 0 | 0 |
| dmf_kuzmin2020 | 632,998 | 658 | 632,340 | 342 | 488 |
| tmf_kuzmin2020 | 301,798 | 317 | 301,481 | 166 | 238 |
| dmi_kuzmin2020 | 632,797 | 658 | 632,139 | 342 | 488 |
| tmi_kuzmin2020 | 301,798 | 317 | 301,481 | 166 | 238 |

  658 + 342 = 1,000 and 317 + 166 = 483, the records the verifier counted. Every other kept record serializes as before; LMDB keys after the first refused row shift down.

- **Not done here.** The 33 Dmf SDs of 0 (#897 L2 `uncertainty_sanity`) belong to the SD-zero work.
- **Verifier accounting (#980, landed).** `count` (632,998 / 301,798 / 632,797 / 301,798) and `released_values` read the whole release; the same branch sets the counts to the release's less the refused rows (632,340 / 301,481 / 632,139 / 301,481) and drops the released rows naming an ORF in `SGA_SYSTEMATIC_REFUSALS` from the compared multiset.

### Rebuilt stores and L0-L4 verification (2026.10.11)

Rebuilt by slurm array 3825 (`gilahyper_build_dataset_lmdbs_array.slurm`, `--retire-existing --verify`, 16 CPU / 64 G, code snapshot `a41c2628b`; the loader code is identical at the branch head). Each store's LMDB entry count equals its `systematic_name_ledger.json` `n_rows_out`: Smf 472, Dmf 632,340, Tmf 301,481, Dmi 632,139, Tmi 301,481. Verified read-only by slurm job 3855 with this branch's verifiers (PR #980 as landed plus the refused-row accounting, snapshot `a93b2ddab`) through `kg4_pre_build_verification_sweep.py`; tables rendered by `experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables.py` from each store's `preprocess/verification_report.json`.

`current_genome_genes` and `canonical_gene_names` now pass on all five stores (#897 to #900), and so do `count` and `released_values` against the release less the refused rows. The one failing rule left is Dmf's `uncertainty_sanity` (33 released SDs of 0), which belongs to the SD-zero work.

`smf_kuzmin2020`: **PASS**, 14 rules, 0 failing (report written 2026-10-11T02:20:01)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 472 records validated |
| L1 | `count` | PASS | observed 472, expected 472 |
| L1 | `pair_uniqueness` | PASS | 472 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 472 values checked |
| L2 | `se_nonnegative` | PASS | 472 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 472 records |
| L1 | `provenance_gaps` | PASS | 472 documented provenance gaps over 472/472 records; 0 deferred field(s): []; 15104 undeclared None values over 8 carrier fields (top: Compound.inchi x6136, Compound.chebi_id x4248, Compound.smiles x1416, Compound.inc... |
| L1 | `canonical_gene_names` | PASS | 235 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | PASS | 472 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 3304 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 472 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 235 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 235 measured systematic names is a gene of the current genome |

`dmf_kuzmin2020`: **FAIL**, 14 rules, 1 failing (report written 2026-10-11T02:26:30)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 632340 records validated |
| L1 | `count` | PASS | observed 632340, expected 632340 |
| L1 | `pair_uniqueness` | PASS | 632340 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 632340 values checked |
| L2 | `se_nonnegative` | PASS | 632340 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 632340 records |
| L1 | `provenance_gaps` | PASS | 632340 documented provenance gaps over 632340/632340 records; 0 deferred field(s): []; 20234880 undeclared None values over 8 carrier fields (top: Compound.inchi x8220420, Compound.chebi_id x5691060, Compound.smiles x... |
| L1 | `canonical_gene_names` | PASS | 4335 systematic names, one canonical spelling each, each current in the genome; 13 common names the resolver cannot place; 12 are pseudogene loci the genome resolves to themselves |
| L2 | `uncertainty_sanity` | **FAIL** | 33/632340 labeled uncertainties are a sample dispersion of exactly 0 (a pseudocount artifact, not perfect precision) |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 4426380 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 632340 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 4335 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 4335 measured systematic names is a gene of the current genome |

`tmf_kuzmin2020`: **PASS**, 14 rules, 0 failing (report written 2026-10-11T02:30:24)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 301481 records validated |
| L1 | `count` | PASS | observed 301481, expected 301481 |
| L1 | `pair_uniqueness` | PASS | 301481 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | PASS | 301481 values checked |
| L2 | `se_nonnegative` | PASS | 0 values checked |
| L3 | `reference_one` | PASS | reference fitness == 1.0 for all 301481 records |
| L1 | `provenance_gaps` | PASS | 301481 documented provenance gaps over 301481/301481 records; 0 deferred field(s): []; 11154797 undeclared None values over 13 carrier fields (top: Compound.inchi x3919253, Compound.chebi_id x2713329, Compound.smiles ... |
| L1 | `canonical_gene_names` | PASS | 4335 systematic names, one canonical spelling each, each current in the genome; 13 common names the resolver cannot place; 12 are pseudogene loci the genome resolves to themselves |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 2110367 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 301481 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 4335 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 4335 measured systematic names is a gene of the current genome |

`dmi_kuzmin2020`: **PASS**, 23 rules, 0 failing (report written 2026-10-11T02:39:00)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 632139 records validated as GeneInteractionExperiment |
| L0 | `reference_structural` | PASS | 632139 records validated as GeneInteractionExperimentReference |
| L1 | `count` | PASS | observed 632139, expected 632139 |
| L1 | `pair_uniqueness` | PASS | 632139 unique (strain, environment) records, one each |
| L1 | `interaction_order` | PASS | all 632139 records perturb 2 distinct genes at graph level 'edge' |
| L2 | `score_finite` | PASS | 632139 interaction scores are finite numbers |
| L2 | `p_value_in_unit_interval` | PASS | 632139 stated p-values lie in [0, 1] (0 records state none) |
| L2 | `released_values` | PASS | the 632139 stored (score, p-value) pairs are the released multiset (the 'digenic' rows of Tables S1 and S3 except those naming an ORF R64-4-1 retired or merged (SGA_SYSTEMATIC_REFUSALS)) |
| L3 | `reference_zero` | PASS | every one of the 632139 reference scores is 0 with no p-value, at the experiment's graph level |
| L3 | `reference_environment` | PASS | every one of the 632139 references is measured in its experiment's own environment |
| L3 | `signed_unclamped` | PASS | 336042 aggravating and 296090 alleviating scores, 7 at exactly zero |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L1 | `provenance_gaps` | PASS | 632139 documented provenance gaps over 632139/632139 records; 0 deferred field(s): []; 22757004 undeclared None values over 12 carrier fields (top: Compound.inchi x8217807, Compound.chebi_id x5689251, Compound.smiles ... |
| L1 | `canonical_gene_names` | PASS | 4335 systematic names, one canonical spelling each, each current in the genome; 13 common names the resolver cannot place; 12 are pseudogene loci the genome resolves to themselves |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 4424973 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 632139 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 4335 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 4335 measured systematic names is a gene of the current genome |
| L4 | `fitness_companion_containment` | PASS | every one of the 632139 (strain, environment) keys is a record of dmf_kuzmin2020 |

`tmi_kuzmin2020`: **PASS**, 23 rules, 0 failing (report written 2026-10-11T02:44:12)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 301481 records validated as GeneInteractionExperiment |
| L0 | `reference_structural` | PASS | 301481 records validated as GeneInteractionExperimentReference |
| L1 | `count` | PASS | observed 301481, expected 301481 |
| L1 | `pair_uniqueness` | PASS | 301481 unique (strain, environment) records, one each |
| L1 | `interaction_order` | PASS | all 301481 records perturb 3 distinct genes at graph level 'hyperedge' |
| L2 | `score_finite` | PASS | 301481 interaction scores are finite numbers |
| L2 | `p_value_in_unit_interval` | PASS | 301481 stated p-values lie in [0, 1] (0 records state none) |
| L2 | `released_values` | PASS | the 301481 stored (score, p-value) pairs are the released multiset (the 'trigenic' rows of Tables S1 and S3 except those naming an ORF R64-4-1 retired or merged (SGA_SYSTEMATIC_REFUSALS)) |
| L3 | `reference_zero` | PASS | every one of the 301481 reference scores is 0 with no p-value, at the experiment's graph level |
| L3 | `reference_environment` | PASS | every one of the 301481 references is measured in its experiment's own environment |
| L3 | `signed_unclamped` | PASS | 140719 aggravating and 160757 alleviating scores, 5 at exactly zero |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in si1.md |
| L1 | `provenance_gaps` | PASS | 301481 documented provenance gaps over 301481/301481 records; 0 deferred field(s): []; 10853316 undeclared None values over 12 carrier fields (top: Compound.inchi x3919253, Compound.chebi_id x2713329, Compound.smiles ... |
| L1 | `canonical_gene_names` | PASS | 4335 systematic names, one canonical spelling each, each current in the genome; 13 common names the resolver cannot place; 12 are pseudogene loci the genome resolves to themselves |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 2110367 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 301481 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 4335 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 4335 measured systematic names is a gene of the current genome |
| L4 | `fitness_companion_containment` | PASS | every one of the 301481 (strain, environment) keys is a record of tmf_kuzmin2020 |
