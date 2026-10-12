---
id: eoj643nfrwr9o28ok3ioek7
title: Hu2007
desc: ''
updated: 1791736566552
created: 1791736566552
---

## 2026.10.11 - Hu 2007 TF-deletion transcriptomes: 263 strains loaded as released X scores

Loader: `torchcell/datasets/scerevisiae/hu2007.py` (`MicroarrayScoreHu2007Dataset`, dev store `data/torchcell/microarray_score_hu2007`). Source: Hu Z, Killion PJ, Iyer VR. Nat Genet 2007;39:683-687, doi:10.1038/ng2012, PMID 17417638, group-library citation key `huGeneticReconstructionFunctional2007` (Zotero item `5P8BEQZ4`). Yeast expansion schedule row 10. Verifier: [[torchcell.verification.expression_score]]. Adapter: `torchcell/adapters/hu2007_adapter.py` with `conf/microarray_score_hu2007_adapter.yaml`. Schema: #1022. Every count below that is not a verifier row comes from `experiments/036-dataset-fixes-before-kg-build/scripts/hu2007_release_inventory.py` (`results/hu2007_release_inventory.json`).

### What the release is

The GEO accession is in the Methods: "Expression data from this study have been deposited in the Gene Expression Omnibus (GEO) database under the series accession number GSE4654." GSE4654 holds 269 samples on platform GPL3588 ("UT Austin Iyer yeast 6.4k ORF-only cDNA platform", spotted cDNA, 6,429 elements), one sample per strain. Each sample releases two columns, `#VALUE = X score` and `#P = p value`, and the series summary says "Data values are X scores and corresponding p-values derived using an error model." Every sample is ch1 = Cy5 = the deletion and ch2 = Cy3 = "WT mid-log growth"; no dye swap is released.

The X score is the Hughes 2000 error model's confidence-weighted measure of the ratio, averaged over the replicate arrays: "To identify significantly regulated target genes, we adapted an error model5, which generates a normalized, confidenceweighted measure of each ratio called the $X$ score (Supplementary Methods)." and "We used normalized microarray data to generate $X$ scores in MATLAB and performed weighted duplicate averaging with custom Java software." Measured on the release: 1,707,575 released scores, and `max |p - 2 * (1 - Phi(|X|))| = 5.0e-7` over all of them, the six-decimal rounding of the P column, so the p-value is the two-sided normal tail of X. No log ratio and no channel intensity is released. "Primary data are also available at http://www.iyerlab.org/tfnetwork": that URL now 301-redirects to a faculty page, and the Wayback CDX index for `iyerlab.org` holds no `tfnetwork` capture (queried 2026-10-11). The per-array data went to the lab's Longhorn Array Database, which is not reachable.

Why a new phenotype class (#1022): `MicroarrayExpressionPhenotype`'s label is documented as `log2(sample/reference)` and its linear `expression` field is required. An X score is a significance statistic, not a log ratio, and the release carries no signal, so storing it there would assert a quantity the release does not carry. `MicroarrayExpressionScorePhenotype` is a sibling leaf that holds exactly the released pair (`expression_score`, `expression_score_p_value`) with `measurement_type`, `reference_basis`, `array_platform`, a strain-level `n_replicates` and the optional `harvest_optical_density`.

### Sourced values

All quotes are verbatim from the library mirror; `paper.md` sha256 `74ada81539e7d599f36972a45703dd2b843552328c0b7e389b31728d639555ab`, `si/si8.md` (Supplementary Methods) sha256 `1e8111da0f5452354b5162ae62eecebeb36094fd3ffa72bb74a04f2bc0982f59`, `si/si2.md` (Supplementary Table 1) sha256 `f0c59f2b5473b6a1ffcdcd67a5228984902b8296c6c353dcc25620b0e5f848c3`. The verifier re-reads all 16 against those bytes.

| value | quote | where |
|---|---|---|
| strain background BY4741 | "Yeast knockout strains were obtained from Open Biosystems and were derived from a BY4741 (MATa his3D1 leu2D0 met15D0 ura3D0) parent26." | paper.md, Methods 'Strains' |
| medium YPD, mid-log | "For each of the 263 nonessential transcription factors, knockout strains were grown in YPD medium to mid-log phase and then harvested." | paper.md, Methods 'Growth conditions' |
| n_replicates = 2 | "Each experiment was independently carried out at least twice." | paper.md, Methods 'Growth conditions' |
| replicates are biological | "The repeatability of each gene's measured expression change across multiple independent biologically repeated experiments is the strongest input to the assignment of a confidence value." | si/si8.md, 'Significant Target Determination' |
| reference culture | "A wild-type strain was cultured together with each batch of mutants as a growth control." | paper.md, Methods 'Growth conditions' |
| reference design | "Because we used a common reference RNA design for all microarray hybridizations, we calculated the true relative expression level of each gene in the mutant relative to the wild-type, normalizing for growth and reference batch effects29." | paper.md, Methods 'Data analysis' |
| dyes | "cDNA samples from wild-type controls were generally labeled with Cy3, and samples from knockout strains were generally labeled with $\mathrm { C y } 5$ ." | paper.md, Methods 'Expression profiling' |
| measurement type | the X score and duplicate-averaging sentences above | paper.md, Methods 'Data analysis' |
| p-value | "A $P$ value for every gene in the resulting data set was calculated in MATLAB based on the averaged $X$ scores." | paper.md, Methods 'Data analysis' |
| harvest OD | "Average OD ofreplicates" (column header) and "Supplementary Table 1 contains additional information on the growth rate of all the strains used." | si/si2.md; paper.md |
| TetO7 treatment | "For six essential transcription factors, $T e t O _ { 7 }$ -promoter regulated mutant strains12 were grown in YPD medium in the presence or absence of $1 0 ~ \mathrm { \mu g / m l }$ doxycycline (Sigma) for $1 4 { - } 1 6 \mathrm { ~ h ~ }$ and then harvested." | paper.md, Methods 'Growth conditions' |
| heat shock | "For 22 of these transcription factors implicated in the heat shock response, deletion strains were grown in YPD medium to mid-log phase and then were subjected to a $1 5 \mathrm { - m i n }$ heat shock at $3 9 ~ ^ { \circ } \mathrm { C }$ before harvest." | paper.md, Methods 'Growth conditions' |

Temperature is not stated. The Methods and the Supplementary Methods were read in full; the only temperature is the heat-shock arm's 39 C shift, and the Methods defer only RNA extraction, array manufacture and hybridization (refs 7 and 27), not growth. `Environment.temperature` is therefore `None` with a `not_reported_by_primary` ProvenanceGap.

### Records

Supplementary Table 1 (the pypdf text layer of `si/si2.pdf`, 292 rows) lists 263 "mig-log phase" strains, 6 "mig-log phase, tetracycline treatment" strains, 22 "mig-log phase, heat-shock" profiles and one "mig-log phase, tetracycline treatment, heat-shock" profile. GSE4654's 269 samples are exactly the 263 mid-log strains plus the 6 tetracycline strains, all "mid-log growth".

- **Served: 263 records**, one per mid-log deletion strain: `KanMxDeletionPerturbation` in BY4741 (common name from the genome's standard name), `YPD_LIQUID`, 1,559,117 (score, p) pairs over 5,997 keyed elements, `n_replicates = 2`, the strain's harvest OD from Table 1. The reference is the neutral score 0 for every element the record scores, with no p-value (259 distinct reference phenotypes: the key sets differ by the released null spots).
- **Refused: 6 records** (`teto7_promoter_strain_reference_and_treatment_unresolved`, #1024): REB1, HSF1, ABF1, MCM1, RAP1, GCR1. They are TetO7 promoter-replacement strains, not deletions; "in the presence or absence of 10 ug/ml doxycycline" does not say which culture is the sample or whether the reference is the wild type or the untreated strain. Measured own-element X scores: GCR1 -2.04, RAP1 -1.70, REB1 -1.09, HSF1 -0.84, ABF1 +0.80, MCM1 +5.03; MCM1's is the opposite of a shut-off. Doxycycline also has no compound-table row.
- **Not released: 23 heat-shock profiles** (22 deletions and TetO7-HSF1) are listed in Table 1 and absent from GSE4654; `preprocess/dropped_records.json` lists them under `not_released`.

### Array elements

GPL3588's ID column is 2006-era names ("May not match current SGD annotations"), resolved through `SCerevisiaeGenome.resolve_gene_name`. 5,997 of 6,429 are keyed: 5,878 current genes, 29 renamed ORFs keyed by their current id (for example `YBL101W-A` -> `YBL100W-A`, `SNR44` -> `YNCL0046W`) and the valid non-gene loci by their own id. Dropped (`preprocess/array_elements.json`): 412 with no R64 locus (348 `i`-prefixed and 29 `Int`-prefixed intergenic elements, 33 old-style Ty and LTR names, and the ORFs YCL006C and YIL080W), and 20 whose resolution lands on an element the array carries directly (for example `YOR240W` -> `YOR239W`), which would score one locus twice. Released null spots are not keys: over the 263 kept samples, 9 values are null in both columns and 18,645 carry `p = 1` beside a null score (3,180 and 18,646 over all 269 samples).

### Owner decisions taken by recommendation

- **A sibling phenotype class, not `MicroarrayExpressionPhenotype`** (#1022): the release is a score with a p-value, and the log-ratio class would have to carry it under a field documented as a log ratio, beside a required signal the release does not have.
- **`n_replicates = 2`**, the conservative lower end of "at least twice"; no per-strain or per-gene count is released.
- **The six TetO7 strains are refused** (#1024), for the reasons above.
- **FLO8 (YER109C) is a deletion record**: it is a blocked reading frame in S288C, a valid R64 locus by its own id, retained as Cachera 2023 and Ohya 2005 retain such loci. L4 names it as the one non-gene deleted locus.
- **A null score beside `p = 1` is not a key**: there is no score to key the p-value to.
- **Harvest density is `harvest_optical_density`, not an OD600**: Table 1 does not state the wavelength.
- **Aerobicity is the schema default** (`aerobic`); the paper states none and the field cannot be gapped.

### Deposits

Raw mirror `$DATA_ROOT/torchcell-raw/huGeneticReconstructionFunctional2007/`, `manifest.json` written by `deposit_raw_mirror`:

- `data/GSE4654_family.soft.gz`: sha256 `0e8bff679733a659b29c0828812ea47ccafd5230d202e0a71813a5d9c05082b4`, 28,122,230 bytes, `direct_url` `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE4nnn/GSE4654/soft/GSE4654_family.soft.gz`, retrieved 2026-10-11 by `torchcell.literature.retrieve.direct_url` (a curl retrieval the same day gave the same sha256).
- `si/si2.pdf`: sha256 `3b9d38d2399979fb8dcd33a041a275342eb7cc7e9c5b12044dd1c290059369ce`, `springer_esm` (Springer ESM MOESM20), the same bytes the library mirror holds.
- `si/si2.txt`: sha256 `22f64c0635f85220a878e99e1013b526e2e6def4b430aa364878b09a9f861a8d`, the pypdf 6.19.0 text layer of `si/si2.pdf`.

No sequence data; nothing in the genomes tier.

### Schema impact

`scripts/schema_impact_check.py --base origin/main`: 8 changed symbols (the three new classes added; `PhenotypeType`, `ExperimentType`, `ExperimentReferenceType`, `EXPERIMENT_TYPE_MAP`, `EXPERIMENT_REFERENCE_TYPE_MAP` gained a member), 0 breaking, 19 dataset classes in 9 groups stale through the `ExperimentType` union. Additive; the KG 4.0 full rebuild covers it.

### L0-L4 verification

`python -m torchcell.database.build_dataset_lmdb --dataset MicroarrayScoreHu2007Dataset --retire-existing --verify` (the loader's `verify_build`; registry `EXPRESSION_SCORE_DATASETS`, family runner `run_expression_score`). Report: `$DATA_ROOT/data/torchcell/microarray_score_hu2007/preprocess/verification_report.json`. The dev store was rebuilt from the clean tree at `f6e894ddf` after the rebase onto `628a0412a` (12 s; build manifest `torchcell_dirty: false`, absent from `--list-stale`), and its `--verify` pass reproduced the table below row for row.

| level | rule | result |
|---|---|---|
| L0 | structural | 263 records validated |
| L1 | count | 263 of 263 (Table 1 mid-log strains in GSE4654) |
| L2 | score_finite | 1,559,117 scores finite |
| L2 | p_value_in_unit_interval | 1,559,117 p-values in [0, 1] |
| L2 | p_value_keys_match_scores | every record keys its p-values like its scores |
| L2 | released_value_fidelity | 1,559,117 stored (score, p) pairs equal the GSE4654 release exactly, re-read from the mirror |
| L3 | reference_score_zero | every reference score is 0, no reference p-value |
| L3 | deletion_lowers_score | median deleted-gene score -1.962; 0.837 negative over 257 (6 deleted genes have a null spot in their own sample) |
| L3 | p_value_is_two_sided_normal | max deviation 5e-7 over 1,559,117 pairs (tolerance 5.01e-7) |
| L3 | sourced_value_audit | 16 quotes verbatim in their pinned mirror files |
| L4 | deleted_gene_containment_sgd | 263 of 263 deleted ORFs are R64 loci (1 non-gene: YER109C) |
| L4 | matched_profile_agreement_microarray_kemmeren2014 | 255 shared strains: median matched r 0.097 vs mismatched 0.031, one-sided Mann-Whitney p = 3e-23 (alpha 1e-3) |

### Kemmeren 2014 cross-check

255 of the 263 strains are also deletion strains of the built Kemmeren 2014 store (`microarray_kemmeren2014`, log2 ratio vs refpool). For each, Pearson r between Hu's X score and Kemmeren's log2 ratio over the genes both carry (5,566 genes in the union, less every shared strain's deleted gene), against the 64,770 mismatched pairs (a Hu strain against another Kemmeren strain):

| statistic | matched (255) | mismatched (64,770) |
|---|---|---|
| median r | 0.097 | 0.031 |
| q25 / q75 | 0.010 / 0.182 | |
| 95th percentile | 0.376 | 0.162 |

172 of the 255 matched r exceed the mismatched median, 79 exceed the mismatched 95th percentile and 45 the 99th; 54 are negative. Highest: YIR033W (MGA2) 0.623, YDR310C (SUM1) 0.489, YLR403W (SFP1) 0.478, YOL004W (SIN3) 0.478, YPL254W (HFI1) 0.457. The two measurements of one strain are two records, never a duplicate. Hypothesis (untested): the low median reflects that most TF deletions change few genes in either screen, plus the different media (Hu YPD; the Kemmeren loader records SC) and statistics (an error-weighted score against a log ratio); neither factor was measured separately here.

### Candidate gate

`python -m torchcell.candidates gate --row "Hu 2007" --table yeast --phenotype-class MicroarrayExpressionScorePhenotype --write`: G1 to G5 pass, outcome `admissible`; verdict `database/candidates/huGeneticReconstructionFunctional2007.json`. G4-value (`overlap`) was not run: it compares one scalar per record and these records are per-gene maps; the L2 `released_value_fidelity` row compares every stored value with the release instead. The ledger was not appended: `ledger` refuses a second 2026.10.11 section and one already exists.

### Next build

Dev-tree store only. Admission is the KG 4.0 full rebuild (the union growth marks 19 served classes stale). The adapter serves the Kemmeren 2014 graph shape with `microarray expression score phenotype` nodes; it is not in the build-time projection's calibration tables.

## 2026.10.12 - The score leaf under the two-channel parent; the X-to-p conversion recorded (#1022, #989)

- **Placement.** `MicroarrayExpressionScorePhenotype` is now a leaf of `TwoChannelMicroarrayExpressionPhenotype`, beside the served `MicroarrayExpressionPhenotype`, Gasch 2000's `MicroarrayExpressionRatioPhenotype` and Hackett 2020's `InductionTimecourseExpressionPhenotype` ([[torchcell.datamodels.schema]], 2026.10.12). Its fields, validators and serialization are unchanged (measured: identical core schema and JSON schema against main `34e76c559`); the base change moves its fingerprint, so the dev store reads stale until the KG 4.0 wave rebuilds it. The loader did not change.
- **Conversion.** The p-value the release carries is `MicroarrayExpressionScorePhenotype.p_value_of_score(X)` = erfc(|X| / sqrt 2) = 2 (1 - Phi(|X|)). Re-measured on `data/GSE4654_family.soft.gz` (sha256 `0e8bff679733a659b29c0828812ea47ccafd5230d202e0a71813a5d9c05082b4`) with a stand-alone reader: 269 samples, 1,707,575 (X, p) pairs, 21,826 pairs with a null side, max |p - erfc(|X|/sqrt 2)| = 5.0e-7. The L3 row `p_value_is_two_sided_normal` now computes its expected p through that method, so the verifier and the schema hold one formula.
- **Candidate gate.** Re-run at branch commit `fcf97be6a`: G1 to G5 pass, `admissible`, the same outcome and reasons as the stored verdict, which was left unchanged.
- **Per-gene n.** The strain-level `n_replicates = 2` (the conservative lower end of "Each experiment was independently carried out at least twice") stays: the release states no per-gene count, and changing the field's type would be a meaning change on a verified class.

## 2026.10.12 - Released element IDs as keys, no resolution (owner rule)

Owner rule, relayed 2026-10-12: the data is the data; no loader drops a released row on genome or annotation grounds, and no resolution status is stored beside it.

- **Elements.** Every GPL3588 element is now keyed by its released ID, verbatim. The loader no longer resolves against R64, so nothing is dropped or merged: the 412 elements R64 does not carry (348 `i`- and 29 `Int`-prefixed intergenic elements, 33 old Ty and LTR names, YCL006C, YIL080W) and the 20 dubious ORFs a neighbour absorbed keep their own released names. Measured from the release bytes over the 263 kept samples: **6,429 elements keyed, 1,672,173 (score, p) values** (the R64-keyed store held 5,997 and 1,559,117). The 6,429 IDs are unique as released. Null scores stay unkeyed, as before.
- **Strain names.** `perturbed_gene_name` is Supplementary Table 1's gene name, or the ORF where the table names none, instead of the genome's current standard name. Against the current dev store this changes 26 of 263 records: 20 take the ORF (TOD6, EDS1, ERT1, NSI1, VMS1, HEL2, URC2, JHD1, COM2, TOG1, OTU1, ZNF1, ASG1, GSM1, OAF3, TDA9, RCO1, PUL4, USV1, SDD4 are names Table 1 does not give) and 6 take the 2007 name (YER109C FLO8, YGL071W RCS1, YJR122W CAF17, YJR127C ZMS1, YNL139C RLR1, YOR191W RIS1). `systematic_gene_name`, the join key, is unchanged.
- **Gate removed.** The deleted ORF is no longer required to be an R64 locus (all 263 were, so no record moves for that reason). `verify_build` still reads a genome for the L4 `deleted_gene_containment_sgd` row, a measurement, not a filter.
- **Store.** The shared dev store still holds the R64-keyed build; the rebuild runs under slurm into a scratch data root (job 4295, held by a user hold at the time of writing). The Kemmeren 2014 cross-check numbers in the 2026.10.11 section were measured on the R64-keyed store and are re-measured on the rebuilt one.

### Rebuild and L0-L4 at `f92873d72` (slurm job 4347)

Built with `build_dataset_lmdb --verify` into a scratch data root (links to the shared raw mirror, library, genomes tier and the Kemmeren 2014 store, read only), so the shared dev store keeps the R64-keyed build until this lands. `preprocess/array_elements.json`: 6,429 of 6,429 elements keyed, 1,672,173 stored values, 9 released null/null and 18,645 null/`p = 1` (source nulls, not values). Job 4295 had built the R64-keyed loader under the reparented schema at `10974b8f3` first: the same 263 records and the PR #1029 numbers, so the schema change alone moved nothing.

| level | rule | result |
|---|---|---|
| L0 | structural | 263 records validated |
| L1 | count | 263 of 263 |
| L2 | score_finite, p_value_in_unit_interval, p_value_keys_match_scores | 1,672,173 values |
| L2 | released_value_fidelity | 1,672,173 stored (score, p) pairs equal the GSE4654 release exactly |
| L3 | reference_score_zero | 0 for all 1,672,173 reference values, no reference p-value |
| L3 | deletion_lowers_score | median deleted-gene score -1.962, 0.837 negative over 257 deleted genes (6 carry no score in their own record) |
| L3 | p_value_is_two_sided_normal | max deviation 5e-7 over 1,672,173 pairs (tolerance 5.01e-7) |
| L3 | sourced_value_audit | 16 quotes verbatim |
| L4 | deleted_gene_containment_sgd (finding, never a gate) | 263 of 263 deleted ORFs are R64 loci, 1 a non-gene locus, 0 not in R64 |
| L4 | matched_profile_agreement_microarray_kemmeren2014 | 255 shared strains, median matched r 0.096 vs mismatched 0.031, one-sided Mann-Whitney p = 4.1e-23 |

Kemmeren cross-check re-measured on this store by `experiments/036-dataset-fixes-before-kg-build/scripts/hu2007_release_inventory.py` (results JSON updated): 5,548 shared genes (5,566 on the R64-keyed store; the renamed elements keep their 2006 names and no longer meet Kemmeren's current ones), matched r quantiles q05 -0.076, q25 0.011, median 0.096, q75 0.183, q95 0.377; mismatched median 0.031, 95th percentile 0.162; 173 of 255 matched r above the mismatched median, 80 above its 95th percentile.

Counts against the release: 269 GEO samples, 263 stored; the 6 not stored are the TetO7 strains refused under #1024 (reference and treatment unresolved, a field gate the 2026-10-12 rule would also revisit: it is an owner decision filed there, not taken here). Every non-null released (score, p) pair of the 263 is stored.
