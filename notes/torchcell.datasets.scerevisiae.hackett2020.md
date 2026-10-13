---
id: nzz4r8yabieunqsm7phyaau
title: Hackett2020
desc: ''
updated: 1791745449066
created: 1791745449066
---

## 2026.10.11 - Hackett 2020 IDEA: 1,752 arrays of 224 induction time courses, loaded

Loader: `torchcell/datasets/scerevisiae/hackett2020.py` (`InductionTimecourseHackett2020Dataset`, dev store `data/torchcell/induction_timecourse_hackett2020`). Source: Hackett SR, Baltz EA, Coram M, et al., McIsaac RS. Mol Syst Biol 2020;16:e9174, doi:10.15252/msb.20199174, PMID 32181581, citation key `hackettLearningCausalNetworks2020` (Zotero `DTLTQTZA`). Yeast expansion schedule row 9. Verifier: [[torchcell.verification.induction_timecourse]]. Adapter: [[torchcell.adapters.hackett2020_adapter]]. Every count below that is not a verifier row comes from `experiments/036-dataset-fixes-before-kg-build/scripts/hackett2020_release_inventory.py` (`results/hackett2020_release_inventory.json`) or from the build's own `preprocess/release_ledger.json`.

### What the releases are

The paper's Data availability names GEO GSE142864 and the IDEA site ("Microarray data, Gene Expression: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi? acc $=$ GSE142864; http://idea.research.calicolabs.com.", paper.md). Four released artifacts carry the arrays:

| artifact | where | what it holds |
|---|---|---|
| `idea_tall_expression_data.zip` | IDEA site bucket `calico-website-pin-public-bucket/datasets` | 10,454,275 rows = 217 courses, 1,693 arrays, 6,175 genes; both channel medians, `log2_ratio`, `log2_cleaned_ratio`, `log2_noise_model`, `log2_cleaned_ratio_zth2d`, `log2_selected_timecourses`, `log2_shrunken_timecourses` |
| Dataset EV2 (`MSB-16-e9174-s006.xlsx`) | PMC | the 10 validation courses, 80 arrays x 6,175 genes; channel medians, `log2_ratio`, `log2_cleaned_ratio` only |
| Dataset EV1 (`MSB-16-e9174-s005.xlsx`) | PMC | the shrunken layer of the 1,693 site-table arrays as a gene x array matrix ("The full transcriptional dataset is available as Dataset EV1.") |
| GEO GSE142864 | NCBI | 1,768 samples (series matrices: titles, strain, protocols; `GSE142864_RAW.tar` per-spot files) |

The column dictionary of the tall table is the IDEA site's own page (deposited, see below): "log2_ratio ... log2(red / green) subtracting value at time zero", "median of green (reference) channel fluorescence", "median of red (experimental) channel fluorescence", "Non-specific stress response and prominent outliers removed" (cleaned), "estimated noise standard deviation", and for the shrunken column "Most users of the data will want to use this column."

The release reproduces the paper's own counts (verifier L4 `release_matches_paper`): "Collectively, the full cleaned dataset is 1,693 microarrays in 217 experiments, each with 6,175 genes." and "Of the 1.34 million time series, 118,134 (8.8%) have at least one point that passes the full noise model." (Appendix, `si/si1.md`), and "The signals from these 100,036 timecourses were retained" (paper.md): the thresholded layer has 118,134 non-zero series and the selected and shrunken layers 100,036. Every log2 layer is exactly 0 at t = 0 in all 1,693 arrays.

Two release defects, both handled in the open. Excel turned the gene `OCT1` into the date 2019-10-01 in both EV1 and EV2 (one cell per array, 80 in EV2); the TSV keeps the name, so `read_validation` restores it only when the sheet's gene set is the tall table's minus OCT1 plus exactly that date. And EV1 rounds what the TSV writes in full: to 9 decimals, and below about 1e-4 to 3 significant digits in scientific form (`-3.485000141e-05` is EV1's `-3.49e-05`); over all 10,454,275 cells the max |EV1 - tall| is 4.99986e-08, 10,906 cells differ from the 9-decimal rounding, and the zero pattern is identical.

### The record

One `InductionTimecourseExpressionExperiment` per (time course, time point), i.e. one array. 1,752 records: 1,672 arrays of 214 site-table courses and 80 arrays of the 10 validation courses. Arrays per course (strains x time points): 190 courses of 8, 17 of 7, 12 of 6, 2 of 5, 1 of 9, 2 of 11; sampling minutes 0, 2.5, 5, 7.5, 8, 10, 12.5, 15, 18, 20, 30, 45, 60, 90, 100, 120, 180, 290. 215 Table EV1 strains have at least one served course; 211 distinct induced genes; 12 genes or controls were run as more than one course (ARR1, CBF1, DAL80, GCN4, GEV, GLN3, GZF3, MAC1, MET4, RDS2, SWI1, Z3EV), each course its own records through `timecourse_id`.

- **Phenotype** (`InductionTimecourseExpressionPhenotype`, new): primary `log2_cleaned` (relative to the same culture at t = 0), statistic `log2_noise_sd` (the authors' per-observation noise SD; one array per culture and time, so the SE of that observation), plus `sample_channel_intensity` (red, Cy5), `reference_channel_intensity` (green, Cy3: "RNA from references and samples was labeled with Cy3-CTP and Cy5-CTP, respectively."), `log2_ratio_unprocessed`, `log2_thresholded`, `log2_selected`, `log2_shrunken`, and the release's `timecourse_id`, `strain_label`, `experiment_date`, `release_table`. Validation arrays carry typed gaps on the noise SD, the three thresholded layers and the date, because EV2 does not release them. Gene keys are R64-4-1 systematic names through the genome's layered resolver: 1,091 current, 5,079 by standard name, 4 non-gene features (AAD6, CRS5, FLO8, YIL082W-A), 1 retired (AAD16, kept verbatim); `preprocess/gene_keys.json` maps every release name.
- **Reference**: the same culture at t = 0 (its own t = 0 array, every log2 layer 0), on its typed strain background (`StrainReferenceGenome`) in its chemostat (`CultureEnvironment`). 224 references, one per course; `experiment_reference_index.json` is 426 MB because each holds a full array.
- **Genotype**: Table EV1 (`si/si2.xls`, sheet `TableS1.txt`, 216 rows). Each strain string matches exactly one of five templates; the designed edit is an `InduciblePromoterInsertionPerturbation` on the induced gene (new leaf): marker plus promoter (`KanMX-Z3EVpr`, `KanMX-Pgal1`, `KanMX-4ZEVpr`, `URA3::Z3EVpr`) inserted "directly between a target gene’s first methionine residue and its native promoter to prevent the removal of any genomic DNA" (Methods), so SO:0000667 `insertion`, `native_promoter_retained`, `responsive_to` the background activator. The GEV and Z3EV control courses carry no perturbation.
- **Backgrounds** (`StrainBackground`, haploid, mating type a typed gap): DBY12021 (`(Pgal10-1+Gal1)::loxP`, `gal4d::LEU2`, `leu2Δ0`, GEV cassette at leu2; 36 strains), SMY69 (the same plus `met6d::HphMX4`; 6), SMY254 (`ybr032wd::LEU2pr(-314 --> -1)-LEU2`, `leu2Δ0`, ZEV cassette at leu2; 161), the 4ZEV parent of SMY257 (1), and the HAP1+ Z3EV background of SMY2047 and SMY2605-SMY2614 (`ura3Δ0`, `his3Δ1`, `can1Δ::STE2pr-Sp_his5`, `lyp1Δ0`, `HAP1+::pACT1-Z3EV-NatMX::eno2term` at HAP1; 11). GEO states the lineage: "background: s288c".
- **Environment**: `CultureEnvironment` on the IDEA chemostat media of [[torchcell.datamodels.media]] by the release's `restriction` token (P 208 courses, M 6, N 3), 30 C, aerobic, `VESSEL` 500 ml at 300 ml stirred at 400 rpm ("Cultures were aerated with $6 1 /$ min of humidified air at $3 0 ^ { \circ } \mathrm { C }$ , maintained at $3 0 0 ~ \mathrm { m l }$ , and stirred with a magnetic impeller at 400 RPM."), endpoint `fixed_duration`, pre-culture a saturated overnight culture ("Batch growth in the chemostat vessels was initiated from a 1:60 dilution of a saturated overnight culture prior to turning on the chemostat pumps."). `duration_hours` = minutes / 60 after beta-estradiol. The t = 0 array has no inducer ("The t = 0 was taken (between 3 and 5 mL of culture) and vacuum filtered. Estradiol was then added the continuous culture and sampled at various timepoints.", GEO protocol); every later array carries `SmallMoleculePerturbation(estradiol, 1 uM)` ("prior to the addition of $1 ~ \mu \mathrm { M }$ $\beta$ -estradiol to the culture and subsequent sampling.").

### Owner decisions taken by recommendation

1. **Primary layer = cleaned, not shrunken.** The authors recommend the shrunken column to "most users", and EV1 is the shrunken layer; but the cleaned ratio is released for all 1,752 arrays (EV2 has no shrunken layer), it pairs with the authors' per-observation noise SD, and the shrunken zeros are an analysis decision ("all other timecourses were set as invariant (a $\log _ { 2 }$ fold change of zero)"), not measured zeros. The shrunken layer is stored beside it and checked against EV1.
2. **Time is `Environment.duration_hours`.** No served expression class has a time-after-perturbation field, but the served Brunk 2016 time course already puts "hours post-induction" on `Environment.duration_hours`, so no field was added. Minutes are recovered as `duration_hours * 60` (the verifier checks every record against the release's minutes).
3. **A sibling phenotype class with its own graph class**, not a subclass of `MicroarrayExpressionPhenotype`: a subclass makes the served `microarray expression phenotype` node ambiguous under `phenotype_label_map`, whose discriminating properties would also be fields of the subclass.
4. **A new perturbation leaf.** `PromoterReplacementPerturbation` is bacterial-only (locus-tag validator, required `gene_namespace`) and its mechanism is `delins`; the IDEA cassette removes no native DNA.
5. **Background names.** DBY12021's TF strains are "Same as DBY12021 with ..."; the ZEV and met6-delta TF strains are named after the control whose Table EV1 string is exactly their shared part (SMY254, SMY69). That is an identification by string identity, not a stated parentage.
6. **Haploid.** No ploidy is stated; every Table EV1 genotype is one allele per locus with no '/', the haploid convention.
7. **Release strain tokens.** `yRSM<n>` is Table EV1's `SMY<n>` (nine strains; same number and same induced gene, checked; GEO also writes `yRSM`), a trailing `n` is the nitrogen-limited course of a strain (SMY108n, SMY117n, SMY156n), `a`/`b`/`c` several courses of one strain (SMY2266a-c, SMY254a/c), `YukoSMY2047` is SMY2047; the tall table's `CUP2` is Table EV1's `ACE1` (both resolve to YGL166W).
8. **Validation strains** come from Table EV1's ten `Prediction` rows matched by the induced gene; GEO's titles (`SMY2605_HMX1_t0`) confirm each match (verifier L4).
9. **Methionine limitation** (restriction `M`, the six met6-delta courses) is stated only by GEO: "Methionine-limited cultures (McIsaac et al, 2012) were maintained at 7.5 mg/L methionine." Recorded as L-methionine.
10. **Not recorded**: a HAP1 allele (the `+` of `HAP1+::...` stays in the cassette name; whether it marks a repaired allele is not stated), a dilution rate (the Methods give 0.15 to 0.17/h for all cultures; a range has no conservative end for a rate, so it is a typed gap), the spot-level GEO arrays and the three replicate t = 0 hybridizations (#1027).

### Refused

Three courses, 21 arrays, whose strain has no Table EV1 row: `CBF1_SMY22_01/01/2015_P_GEV` (6), `GEV_SMY10_01/01/2015_P_GEV` (8), `MET4_SMY19_01/01/2015_P_GEV` (7); reason `strain_not_in_table_ev1` in the ledger. Their genotypes, the chemostat recipe and the backgrounds' mating type wait on McIsaac 2011/2012/2013 and Saldanha 2004, none mirrored (#1026, by hand).

### Deposits

Raw mirror `$DATA_ROOT/torchcell-raw/hackettLearningCausalNetworks2020/` (`manifest.json`, retrieved 2026-10-11):

| path | sha256 | retrieval |
|---|---|---|
| `data/idea_tall_expression_data.zip` | `2fec89f98c602049f2964ca80821d540b50e579fe799d8694fd742731830accc` | direct_url; bucket md5 `XOJNBALqApEP249WbeOMug==`, generation 1577397324560244, checked |
| `data/idea_website_main.915bfc49.chunk.js` | `41c8070fd8286fea284ca435ba554b8e8d51ccd732f02d05a9b8910b8a8fd8aa` | direct_url (the column dictionary) |
| `data/GSE142864-GPL16244_series_matrix.txt.gz` | `871d1fddba1a32d2152d1dd6117d57b1b00876221a075f72ac54d8a50489cb08` | direct_url |
| `data/GSE142864-GPL20785_series_matrix.txt.gz` | `eed815375636378282ff082f5ecc93309a1f9499393d66f920bdeaacb7ff4f57` | direct_url |
| `si/MSB-16-e9174-s002.xls` (Table EV1) | `22b665e6c40cc36d0f6fdcf9ee51bc18548b72e450745aebba6de8abdfe9e2dc` | pmc_cloud (library record) |
| `si/MSB-16-e9174-s006.xlsx` (Dataset EV2) | `2dd041ba81ed89b000f4198112317f3600aa5035f36892a9b63fb9c6f95751dd` | pmc_cloud (library record) |

No genomes-tier deposit: the genotypes are written against S288C R64 (`sgd_S288C_R64-4-1_20230830`). The 1.4 GB store builds in 6.5 min at 7.4 GB peak RSS, so no slurm job.

### Verification (dev store, 2026-10-11)

| level | rule | result |
|---|---|---|
| L0 | structural | 1,752 records validate as the concrete classes; `EXPERIMENT_TYPE_MAP` reconstructs them |
| L1 | count | 1,752 = 1,752 |
| L1 | array_completeness | every released array stored once, minus the 21 of the 3 refused courses |
| L1 | gene_completeness | every layer of every record carries the 6,175 keys, a bijection with the release names |
| L1 | layer_gaps | a layer is absent exactly when its release does not carry it, as a typed gap |
| L1 | provenance_gaps | 10,688 typed gaps over 1,752 records, 0 deferred fields on records |
| L1 | canonical_gene_names | 211 systematic names, one spelling each |
| L2 | release_fidelity | 84,572,800 stored values equal their released cells bit for bit |
| L2 | value_fidelity, sampling_time, uncertainty_sanity | pass |
| L3 | t0_zero | every log2 layer 0 at t = 0 and in every reference |
| L3 | layer_nesting | shrunken within selected within thresholded; thresholded = cleaned where non-zero |
| L3 | inducer_timing | no estradiol at t = 0, 1 uM after; medium is the course's limitation |
| L3 | induction_upregulates | induced gene up at the last array in 0.995 of 221 courses; median log2 5.92 (60.7-fold); the paper's "median level 53-fold higher than at $t = 0 ~ \mathrm { m i n }$" is at saturation |
| L3 | genotype_matches_release, compound_identity, media_compound_identity, media_membership | pass |
| L3 | sourced_values | 46 quotes verbatim in their pinned bytes (paper, Appendix, GEO gzip, site page, Table EV1 rows) |
| L4 | shrunken_matches_dataset_ev1 | 10,324,600 values within 5e-08 of EV1, zero pattern identical |
| L4 | release_matches_paper | 217 courses, 1,693 arrays, 6,175 genes, 118,134 and 100,036 series |
| L4 | geo_sample_coverage | all 1,752 arrays are GEO samples of the same strain and time (1,768 samples) |
| L4 | induced_gene_containment | 211 induced genes, all S288C genes |

### Schema impact

`scripts/schema_impact_check.py --base origin/main` (after rebasing onto `cc7994a57`): 10 symbols added or gaining members (the leaf, the three timecourse classes, the four unions, the two type maps); 99 dataset entries stale through the `GenePerturbationType` / `ExperimentType` unions, 0 breaking. KG 4.0 is a full rebuild. The graph gains one node class, `induction timecourse expression phenotype`.

### Gate

Candidate verdict `database/candidates/hackettLearningCausalNetworks2020.json`: G1 pass (primary), G2 pass (S288C), G3 pass (68 files), G4 pass, G5 pass, admissible. G3 first refused the deposit: `leading_archive` took the first `PK\x05\x06` in the 368 MB zip, which sits inside its deflate stream at byte 164,465,128; fixed in `torchcell/candidates/gates.py` (an EOCD candidate must agree with the central directory it names). The candidate ledger was not appended: `ledger` refuses a second 2026.10.11 section ("one ledger per day").

## 2026.10.12 - Stored as released: the genome gate is reversed

Owner rule of 2026.10.12 ("the data is the data"; the genome is a separate layer that never gates storage). The loader consults no genome any more.

- Expression keys. Every layer of every array is keyed by the release's own `GeneName` token verbatim (6,175 keys). Until today the build re-keyed 5,082 of them to R64 systematic names through `SCerevisiaeGenome.resolve_gene_name` (5,079 renamed, 4 non-gene features, 1 retired: `AAD16`, which stayed as released). `GeneKey`, `resolve_gene_keys`, `default_resolver` and the dataset's `gene_resolver()` are gone; `preprocess/gene_keys.json` now states the rule and lists the pinned TF table. The `OCT1` restoration stays: Excel turned that cell into a date in both SI workbooks, which is a corruption of the release file, not a genome lookup.
- Induced gene. The release names each strain's induced gene only by a common-name token (Table EV1's genotype string, the tall table's `TF`; `ACE1` in Table EV1 is `CUP2` in the release, `YGR066c` is Table EV1's spelling of the release's `YGR066C`), and `InduciblePromoterInsertionPerturbation.systematic_gene_name` must be an ORF. `RELEASE_TF_TOKENS_WITHOUT_AN_ORF` (213 tokens) is the one remaining crosswalk: the token -> ORF pairs the 2026.10.11 build served, derived once from that build's `gene_keys.json`; the loader makes no genome call. It is pending group 1's released-gene-name leaf (a GenePerturbation child carrying the released symbol with `systematic_gene_name` as a typed gap), after which the token is stored verbatim and the table goes. `perturbed_gene_name` stays the release's TF token.
- Refusal kept. The three time courses on SMY10, SMY19 and SMY22 (21 arrays) have no Table EV1 row; the reference genome needs the strain's background, which has no typed gap, so the refusal stays with its count in `release_ledger.json` (`strain_not_in_table_ev1`). Needs a carrier or an owner call.
- Ledger fields `gene_key_status` and `gene_keys_renamed` are replaced by `n_gene_keys_as_released` (6,175) and `n_tf_tokens_pinned` (213). The verifier (`torchcell/verification/induction_timecourse.py`) reads the release's gene names directly and checks the insertion through the pinned table.
- Record count: unchanged at 1,752 (not served in KG 3.0). Reconciling a release name to the current annotation is a downstream step at adapter or modeling time.

### L0-L4 after the rebuild (slurm 4443_6, `--retire-existing --verify`)

Store `induction_timecourse_hackett2020`: 1,752 records; build manifest commit `7f67ede3e` (dirty: True), built 2026-10-13T00:25:04Z; verdict PASS over 24 rules (0 genome-reference finding(s) reported under the verdict, 0 failing).

| level | rule | result |
|---|---|---|
| L0 | structural | 1752 records validated |
| L1 | count | observed 1752, expected 1752 |
| L1 | array_completeness | 1752 of 1752 released arrays stored once (21 arrays of 3 refused courses excluded) |
| L1 | gene_completeness | every layer of every record carries the 6175 keys, a bijection with the release's gene names |
| L1 | layer_gaps | a layer is absent exactly when its release does not carry it, and then as a typed gap |
| L1 | provenance_gaps | 10688 documented provenance gaps over 1752/1752 records; 0 deferred field(s): []; 30824 undeclared None values over 11 carrier fields (top: Compound.inchi x5032, Compound.chebi_id x4966, Compound.inchikey x3438, Compo... |
| L1 | canonical_gene_names | 211 systematic names, one canonical spelling each (no resolver supplied: spelling checked, annotation not) |
| L2 | uncertainty_sanity | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | compound_identity | environment edits: 1528 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | media_compound_identity | medium components: 66 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | media_membership | 1752 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (3 distinct media) |
| L2 | release_fidelity | 84572800 stored values equal their released cells bit for bit |
| L2 | value_fidelity | every value finite; intensities and noise SDs non-negative |
| L2 | sampling_time | duration_hours x 60 is a released time of the record's course |
| L3 | t0_zero | every log2 layer is 0 at t = 0 and in every reference |
| L3 | layer_nesting | shrunken support within selected within thresholded; thresholded equals cleaned where non-zero |
| L3 | inducer_timing | no estradiol at t = 0 or in a reference, 1 uM estradiol after; medium is the course's limitation |
| L3 | induction_upregulates | the induced gene is up at the last array of 0.995 of 221 courses (median log2 5.92, 60.7-fold; the paper's median saturation is 53-fold) |
| L3 | genotype_matches_release | every course's genotype is the release's induced gene (none for a control) |
| L3 | sourced_values | 46 quotes verbatim in their pinned sources |
| L4 | shrunken_matches_dataset_ev1 | 10324600 stored shrunken values match Dataset EV1 within 5e-08 (max \|diff\| 5.00e-08), zeros where EV1 has zeros; EV1 holds 1693 arrays |
| L4 | release_matches_paper | the release reproduces 217 courses, 1,693 arrays, 6,175 genes, 118,134 thresholded and 100,036 selected and shrunken time series |
| L4 | geo_sample_coverage | all 1752 arrays are GEO GSE142864 samples of the same strain and time (1768 GEO samples) |
| L4 | induced_gene_containment | all 211 induced genes are S288C genes |

The store built in 154 s on commit `7f67ede3e`; the manifest reads `torchcell_dirty: True` because the Puddu 2019 note was being edited in the same worktree during the array (notes only, no code). Every expression layer is keyed by the release names and the pinned induced-gene table verifies against them; no genome-reference finding.
