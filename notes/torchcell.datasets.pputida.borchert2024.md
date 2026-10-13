---
id: heop40a145ih4q1uecpe5fe
title: Borchert2024
desc: ''
updated: 1791383159292
created: 1791383159292
---

## 2026.10.07 - The KT2440 RB-TnSeq compendium loader and the superset finding

Source: `torchcell/datasets/pputida/borchert2024.py` (`RbTnseqBorchert2024Dataset`)
Tests: `tests/torchcell/datasets/pputida/test_borchert2024.py`
Plan: [[plan.bacteria-ontology-genome]] section 4 (row 8 of the fifty, status
`aggregation`). Shared skeleton: [[torchcell.datasets.bacteria_common]]. Schema layer:
[[torchcell.datamodels.bacterial-perturbation-ontology]]. Media: [[torchcell.datamodels.media]].

Borchert et al. 2024 (mSystems 9:e00942-23, doi 10.1128/msystems.00942-23, citation key
`borchertMachineLearningAnalysis2024`) compiled 332 RB-TnSeq samples of the KT2440
`Putida_ML5` library into one 4,732-gene matrix and decomposed it with ICA into 84
fModules. The matrix is the superset of the other KT2440 RB-TnSeq rows, so it is loaded
here and every record names the study that first reported its sample.

### Source and raw mirror

The paper's Data Availability: "Gene fitness values, associated statistics, and metadata
for each sample are available at https:// github.com/beckham-lab/fModule." The repository
holds one file.

| item | value |
|---|---|
| file | `fModule_Metadata.xlsx`, sheets `metadata`, `fitness_measurements`, `T-like_statistics` |
| retrieval | `torchcell.literature.retrieve.direct_url("https://raw.githubusercontent.com/beckham-lab/fModule/30eaef39a4609335f0f10c2d25a8eb69d426b0ce/fModule_Metadata.xlsx")` |
| commit | `30eaef39a4609335f0f10c2d25a8eb69d426b0ce` (2023-09-13), git blob `bce51f3614fe4146b036f69edd5339cf78e042e7` |
| sha256 | `4d649385ac06684482396a125f135df22a2a5060da73485b2cd14468f8cc8be1`, 23,261,645 bytes |
| mirror | `$DATA_ROOT/torchcell-raw/borchertMachineLearningAnalysis2024/data/fModule_Metadata.xlsx` + `manifest.json` (`RetrievalMethod.direct_url`, `last_check` matches on 2026-10-07) |

`deposit_raw_mirror()` runs the recorded retrieval, refuses bytes that do not hash to the
pin, keeps a matching mirror file and refuses a differing one. The paper's second
repository, `fModules/putida-code` (`data/raw_data/rb_tn_Seq_332.csv`, commit
`f68ee007`), carries the identical fitness matrix (measured: max absolute difference 0.0
over 4,732 x 332 values) but no t statistics, so it is not mirrored. The live Fitness
Browser answers scripts with a Cloudflare 403 and is not used.

### The superset finding

Measured from the release's `metadata` sheet (sets, persons, media, condition columns)
against the mirrored source papers. A sample is attributed to a study only when that
study's mirrored Methods or Results name its condition; everything else is attributed to
the compendium itself.

| source study (DOI) | row of the fifty | samples | conditions | sets | evidence (verbatim) |
|---|---|---|---|---|---|
| Thompson 2020, fatty acids and alcohols (10.1128/AEM.01665-20) | 28 | 46 | 23 | set15, set16, 10 of set12 | "Global fitness analyses of transposon libraries grown on 13 fatty acids and 10 alcohols"; the alcohol list names set12's 1,2-propanediol, 1,4-butanediol, 1,5-pentanediol, 2-methyl-1-butanol |
| Schmidt 2022, nitrogen (10.1128/aem.02430-21) | 24 | 123 | 71 | set27, set28, 6 of set29, 19 of set10 | "52 different nitrogen containing compounds. To assay amino acid biosynthesis, 19 amino acid drop-out conditions were also tested. From these 71 conditions" |
| Borchert 2023, lignin tolerance (10.1016/j.ymben.2023.04.007) | 29 | 42 | 13 | 36 of set100, 6 of set101 | the Methods' list of 11 stressors plus "nothing" on M9 + 20 mM glucose, and "30 mM protocatechuate" run "on a separate day" with its own glucose control |
| compendium only, attributed to Borchert 2024 (10.1128/msystems.00942-23) | 8 | 121 (79 kept) | 50 | set1, set5, set6, set7, set8, set9, 7 of set10, 22 of set12, 2 of set29, 3 of set100, 33 of set101 | "The Fitness Browser ... was used to obtain data for 254/332 of the samples" |

Every condition of the three subsumed rows is present: Thompson 2020's 23 (13 fatty acids
plus 10 alcohols, two samples each), Schmidt 2022's 71 (33 + 16 + 3 = 52 nitrogen
compounds plus 19 drop-outs; the sample counts match the quoted numbers exactly), and
Borchert 2023's 11 stressors, the protocatechuate enrichment and the glucose control. What
the compendium does NOT carry of those rows: any gene that lacked a value in any of the
332 samples ("In instances where gene fitness data for a particular gene did not exist
across all 332 data sets, the gene was eliminated"), so condition-specific genes of the
primary papers are absent.

Two corrections to the plan's list of subsumed rows, both measured:

- **Price 2018 (row 21) has no KT2440 experiment.** Its Table S5 lists experiments for 32
  organisms with no `Putida` orgId, and Table S14 lists 32 bacteria with no P.
  putida (`priceMutantPhenotypesThousands2018/si/si3.xlsx`, sha256 `e5dbf3d5...`; pinned
  by `test_price2018_carries_no_kt2440_experiment`). Nothing of Price 2018 is subsumed here.
- **Royet 2025 (row 47) is not in the compendium**: it is non-barcoded mariner Tn-seq on a
  different pool, not `Putida_ML5`.

The compendium-only samples, with UNTESTED candidate sources (person, set and condition
columns only; none of these papers is mirrored, so none becomes a record's publication):

| set | kept | conditions | hypothesis, untested |
|---|---|---|---|
| set5 | dropped | glucose, levulinic acid, vanillin, p-coumarate | Rand 2017 (ref 88, the levulinic-acid paper that built Putida_ML5) |
| set1 | dropped | glucose, 4-hydroxyvalerate, acetate | none |
| set6 | 5 | glucose, glucuronate, ferulate, valerolactam, benzoate | Thompson 2019 valerolactam for 2-piperidinone (row 56) |
| set7 | 7 | GABA, 5-aminovalerate, D- and L-lysine, valerolactam | Thompson 2019 lysine (ref 20, row 52) and valerolactam |
| set8, set9 | dropped | bioreactor and adaptation series | Eng 2021 (ref 37) |
| set10 | 7 | 20-amino-acid control, Arg, Glu, His, alpha-ketoglutarate, butyrate, succinate as carbon | none |
| set12 | 22 | aromatics, glucose, Glu, Lys, phenylacetate (deep-well, 1% DMSO) | Incha 2020 (ref 21) |
| set29 | 2 | D-galacturonate | none |
| set100 | 3 | glucose + 200 mM muconate | none |
| set101 | 33 | glucose + 300 mM muconate, beta-ketoadipate, levulinate; eight sole aromatic and branched carbon sources | Borchert 2022 (ref 38) or the 2024 deposit PRJNA1011287 |

For set100 and set101, NCBI BioSample descriptions (fetched 2026-10-07 through E-utilities,
not mirrored, so stated here as retrieval evidence and never quoted on a record) show that
PRJNA856070 (Borchert 2023's SRP385031) holds set100's twelve non-muconate conditions and,
in a second batch dated 2021-12-07, glucose, 300 mM muconate, 30 mM protocatechuate, 125
mM beta-ketoadipate and 125 mM levulinate. PRJNA809672 holds time-zero comparisons and
M9 with 10 mM ferulate; PRJNA1011287 holds 13 triplicate groups with no condition
description.
Borchert 2023's paper reports only the protocatechuate condition of that second batch, so
muconate, beta-ketoadipate and levulinate stay compendium-only.

Net new relative to the fifty's rows: 79 kept samples (373,828 records) in 41 conditions,
plus the 42 samples dropped below.

### Phenotype: why not FitnessPhenotype

RB-TnSeq gene fitness is signed: "The fitness data are normalized so that the typical gene
has a fitness of zero" (Wetmore 2015). `FitnessPhenotype.validate_fitness` clamps every
value at or below 0 to 0.0, and the fitness verifier requires a reference of 1.0, so the
plan's mapping would erase every defect (fitness here ranges from -18.66 to 10.62). Records
are `BacterialEnvironmentResponseExperiment` with
`EnvironmentResponsePhenotype(measurement_type=log2_ratio,
assay_type=pooled_competitive_growth_barcode)`, as the Hillenmeyer HIP/HOP loader does. The
Wetmore 2015 loader (branch `feat/ecoli-wetmore2015`) reached the same decision
independently, and its gene-level unit and `n_samples` gap are matched here. The reference
is the typical gene of the same sample: `environment_response = 0.0`, `screen_id` = the
sample.

Definitions, verbatim:

- Strain and gene fitness (Wetmore 2015): "Roughly, strain fitness is the normalized
  $\log _ { 2 }$ ratio of counts between the treatment sample (i.e., after growth in a
  certain medium) and the reference “time-zero” sample. Gene fitness is the weighted
  average of the strain fitness, and a t score is computed based on the consistency of the
  strain fitness values for each gene." Thompson 2020 and Schmidt 2022 state the same and
  defer to Wetmore ("A more detailed explanation of calculating fitness scores can be found
  in a previous study by Wetmore et al. (40).").
- The t statistic (Wetmore 2015): "(iv) $\pmb { t }$ -like test statistic. To estimate the
  reliability of the fitness measurement for each gene $f ,$ we use a moderated $t$
  statistic:", $t = f / \sqrt{\sigma^2 + \max(V_e, V_n)}$, "where $\sigma$ is a small
  constant (we use 0.1)".
- Beckham sets (Borchert 2023): "Transposon insertion counts were not trimmed from gene
  ends for fitness calculations. Gene fitness was calculated as the weighted average of the
  strain fitness for all transposon insertions at that locus and normalized by subtracting
  the median unnormalized fitness within a 251 gene sliding window."

### Uncertainty, and the Beckham t column

The release carries t, not an uncertainty, so `environment_response_uncertainty` is a typed
gap (`not_reported_by_primary`) on every record. The Fitness Browser's standard error is
defined by Price 2018 (`priceMutantPhenotypesThousands2018/paper.md`, sha256
`f3443cdc...`): "The standard error is the maximum of two estimates. The first estimate is
based on the consistency of the fitness for the strains in that gene. The second estimate is
based on the number of reads for the gene." The Price 2018 E. coli loader (PR #725) measured
`t = f / sqrt(0.1^2 + max(se_obs, se_naive)^2)` to within 6.4e-13 over 613,818 values, which
is Wetmore's t with $V = se^2$. This release carries no standard-error columns, so for a
Fitness Browser sample the standard error is recovered as
$se = \sqrt{(f/t)^2 - 0.1^2}$ and stored as the DERIVED `environment_response_se`.

Both columns are rounded to three decimals. With $q = |f/t|$ and the first-order rounding
bound $\rho = 0.0005/|f| + 0.0005/|t|$, the relative error of $se$ is
$\rho\, q^2 / (q^2 - 0.1^2)$, which grows without limit as $q$ approaches the 0.1 floor. The
standard error is stored when that bound is at most 0.05: 867,978 of the 1,003,184 kept
Fitness Browser records (86.5 %); the other 135,206 carry a gap on `environment_response_se`.

**The identity re-checked in the build.** Without the standard-error columns the identity
cannot be tested directly, but it forces $|f/t| \ge 0.1$. `check_identity_floor` refuses a
build in which any kept Fitness Browser pair falls below 0.1 even at its rounding's upper
bound: 1,000,314 nonzero pairs checked, 0 breaches. Over all 1,201,928 Fitness Browser
values of the release, 680 have $|f/t| < 0.1$ and every one is within its rounding.

An earlier build on this branch stored $|f/t|$ itself, i.e.
$\sqrt{0.1^2 + se^2}$, which keeps Wetmore's normalization constant in the number and so
overstates the standard error; that build was deprecated
(`/scratch/projects/torchcell-deprecated/2026-10-07_114653__processed`) and rebuilt.

**Measured: the 78 Beckham samples' t is not Wetmore's t of the released fitness.** Over
nonzero pairs, sign(f) and sign(t) disagree for 0 % of entries in every Fitness Browser set
and for 49.1 % (set100) and 49.3 % (set101) of entries; the per-sample Pearson correlation
of f with t has median +0.79 to +0.89 in the Fitness Browser sets and -0.53 (set100) and
-0.44 (set101) in the Beckham sets. The fitness sign is the conventional one (hisA, serA and
nadC are strongly negative on minimal glucose in both sources, e.g. nadC -4.81 in
`set100IT008` against -5.23 in `set12IT027`), while t is positive for those genes (nadC t =
10.97). Replicate t values agree with each other (median r 0.85 in set101, 0.87 in set100), and
their magnitudes respect the same 0.1 floor (0 breaches; median $|f/t|$ 0.149 and 0.161), so
it is a real statistic of the same scale whose sign follows an unstated rule. No standard error is derived for the 369,096 Beckham records.
Reproduced on every build in `preprocess/uncertainty.json` (`t_sign_agreement_by_set`).

`n_samples` and `sample_unit` are gaps (`not_carried_by_curation`): a gene fitness averages
a per-gene number of strains that the release does not carry. Replication lies across
records: Thompson 2020 and Schmidt 2022 state biological duplicates, Borchert 2023
biological triplicates, and Borchert 2024 "a mixture of duplicate or triplicate samples";
"Biological and technical replicates present in the data were not averaged".

### Strain, genome and identifiers

`REFERENCE_STRAIN = "KT2440"`; `pputida_genome` is injected by the build entry points.
Strain, verbatim: "All RB-TnSeq data in the ICA data set were generated with a previously
described, randomly barcoded transposon mutant library in P. putida KT2440 (Putida_ML5)
(88)." `genome_reference = assembly_reference("KT2440")`: `pputida_KT2440_ASM756v2`,
`GCA_000007565.2`.

`reconcile_locus_tags` on the 4,732 `locusId` values:

| status | count | layer |
|---|---|---|
| current | 4,732 | locus tag 4,732 |
| renamed, pseudogene, retired, ambiguous | 0 | |

0 remapped, 0 kept on collision, 0 outside the `pputida_kt2440_locus_tag` namespace; the
loader requires 0.99 resolved. `perturbed_gene_name` is the GenBank symbol when the
resolver maps it back to the same locus (1,537 genes), else the locus tag (3,195).

Genotype: one `TransposonInsertionPerturbation` (gene-level aggregate, so `barcode`,
`insertion_position` and `insertion_strand` are `None`), `transposon="mariner (Tc1)"`
("randomly-barcoded mariner (Tc1) transposon insertions", Borchert 2023),
`library_pool` = the release's `mutantLibrary` verbatim: `Putida_ML5` (25 samples) or
`Putida_ML5_JBEI` (307), the regrown JBEI-1 stock of the same library (Thompson 2020).

### Environment

| release medium | samples kept | medium object |
|---|---|---|
| `MOPS minimal media_noCarbon` | 108 | `MOPS_MINIMAL` (the library entry is this Fitness Browser medium, from Price 2018 Table S18) |
| `MOPS minimal media_Glucose_noNitrogen` | 104 | `FB_MOPS_GLUCOSE_NO_NITROGEN`: derived from `MOPS_MINIMAL`, ammonium chloride as dropout, D-glucose with amount `None` (Schmidt 2022 defers to its unmirrored Fig. S1) |
| `M9_medium` | 78 | `BORCHERT2023_M9`: derived from `M9`, Borchert 2023's stated modified M9 (K2HPO4 in OCR and PDF text layer, so the base KH2PO4 is a dropout) |

`condition_1` is the varied carbon or nitrogen source:
`EnvironmentPhysicalPerturbation(factor=carbon_source | nitrogen_source, agent=<compound>,
magnitude=<mM>)`. `condition_2` is a stressor on fixed glucose (`SmallMoleculePerturbation`,
mM), the 1 vol% DMSO of set12, or the 20-amino-acid supplement at 0.5 X (a gapped mixture
compound with `DoseBasis.fixed`, since `ConcentrationUnit` has no `X`) whose `_minus_<AA>`
suffix becomes `nutrient_dropout` of that L-amino acid (Schmidt 2022: "By supplying all but
one of the 20 proteinogenic amino acids"). Temperature 30 C from the release; aerobic;
`duration_hours` and `duration_generations` are gaps (Schmidt harvested at 24 to 72 h by
turbidity, Borchert 2023 at OD600 1.0). Vessel and shaking have no field on `Environment`
and are not stored.

Two label repairs, each recorded in `CONDITION_LABEL_FIXES`: `ÃŸ-ketoadipate` (the UTF-8
bytes of 'ß' read as Latin-1) to `beta-ketoadipate`, and `Protecatechuic acid` to set12's
spelling `Protocatechuic Acid`. Of the 127 condition labels of kept samples, 43 resolve to
an InChIKey and 64 carry `resolved_compound`'s `deferred_pending_source_review` gap; the
compound table is not extended on this branch.

Discrepancies kept visible rather than resolved by guess: Thompson 2020 grows its RB-TnSeq
"in test tubes" at 200 rpm and describes a "modified MOPS" with ten times Neidhardt's trace
metals "when indicated", while the release labels its samples with the Fitness Browser
medium and, for set12, a 96 deep-well plate, 700 rpm and 1% DMSO. The loader follows the
per-sample release label.

### Dropped samples

| rule | samples | records not written |
|---|---|---|
| `medium_not_in_media_library` (`RCH2_defined_noCarbon`, set1 and set5) | 20 | 94,640 |
| `reactor_process_not_representable` (set8 and set9: DO set point, feed regime, sampling time, overlay, adaptation day) | 22 | 104,104 |

Kept: 290 samples x 4,732 genes = **1,372,280 records**.

### Advisory check

The only data advisory in reach is the Fitness Browser compendium page's "Note added
September 10, 2021" (Wetmore raw mirror `data/bigfit/index.html`, sha256 `c4819e45...`):
"Please disregard any of the data from this publication regarding sucrose or D-mannitol."
It concerns Price 2018's E. coli data; the KT2440 release has 0 sucrose or mannitol samples
(measured over both condition columns and both descriptions). No mirrored source paper
carries a withdrawal or correction notice. Thompson 2020 has a published correction (Europe
PMC: PMC8091123, type "correction"); its text is not scriptable (PMC HTML is
challenge-gated, the notice is not in the PMC open-data bucket) and it is not mirrored, so
whether it touches the fitness data is UNKNOWN and listed as an open item.

### Build and verification

`python -m torchcell.database.build_dataset_lmdb --dataset RbTnseqBorchert2024Dataset` on
GilaHyper, 2026-10-07 (the rebuild after the standard-error correction): 1,372,280
records, gene set 4,732, 290 references, 412 s (the first build: 397 s, 6.3 GB peak RSS),
LMDB 4.9 GB (records about 2 KB, no overflow pages). The build manifest reads
`fresh` under `torchcell.provenance.build_manifest`. Each sample's first record is checked
against `_intern_record`'s serialization before the fast path writes the rest.

## 2026.10.08 - The 42 Borchert 2023 samples: subsumption measured in full, and a closable duration gap

`RbTnseqBorchert2023Dataset` landed beside this loader. Measuring the partition between the
two produced two findings about THIS dataset.

**The 42 samples this release attributes to Borchert 2023 are that paper's own cultures,
value for value.** All 13 of Borchert 2023's pairwise comparison sheets were compared
numerically against `fitness_measurements`: each of the 42 (experiment, replicate) arms
equals exactly one sample column here on all 4,732 shared loci, maximum absolute difference
0.000500 over all 42, with the runner-up column never nearer than 1.7885. Borchert 2023
releases full precision and this release rounds to three decimals, so 0.000500 is the
half-unit of that rounding. Corroborated independently by Table S3's BarSeq `IT` index,
which is the index inside each matched sample name on all 42 (`Exp1A` used `IT08`, and the
match is `set100IT008`).

**`duration_hours` is closable for those 42 samples and is still a gap here.** This loader
records `duration_hours` as `not_reported_by_primary` for all 332 samples, correctly: the
release has no growth-time column. Borchert 2023's Table S4 does, per replicate, at the
time of sampling (`10:40`, `10:05`, `10:05` for the glucose reference, and so on for all
14 experiments). `RbTnseqBorchert2023Dataset` stores it. Filling it on the 42 samples here
would change served records, so it is a full knowledge-graph rebuild rather than an
increment, and it was not done. Recorded so the next full rebuild can take it.

The 271 loci Borchert 2023 recovers are the ones this release's own filter eliminated
("In instances where gene fitness data for a particular gene did not exist across all 332
data sets, the gene was eliminated from analysis"). They are disjoint from this store's
4,732 by construction, and the sibling build asserts it against this dataset's LMDB on
every run. Full measurement: [[torchcell.datasets.pputida.borchert2023]].

## 2026.10.10 - Thompson 2019 lysine samples identified by value

Measured, not inferred from names: Thompson 2019 lysine's Table S1 (39 genes x 4 carbon sources) matches compendium samples set7IT062 (D-lysine), set7IT055 (L-lysine), set7IT044 (5-aminovalerate) and set6IT057 (glucose), max abs difference 0.046 against a nearest wrong sample at 0.58. The served store holds all four (18,928 records), so that row needs no loader. This turns the `HYPOTHESES["set7"]` entry into a measurement for three samples and adds set6IT057, which the dictionary does not attribute. Re-attributing them as a `SourceStudy` is an owner decision (store rebuild). The compendium labels set6IT057 a 48-well Tecan microplate; the paper's Methods say 50-ml culture tubes. Details: [[experiments.036-dataset-fixes-before-kg-build.scripts.thompsonMassivelyParallelFitness2019_release_inventory]].

## 2026.10.10 - Royet 2025 measured as independent of the compendium

Measured by `experiments/036-dataset-fixes-before-kg-build/scripts/royetHighThroughputTnSeqScreens2025_release_inventory.py`:
0 of the 332 samples are on LB and 0 name a metal, so no (gene, condition) pair of Royet
2025's LB-vs-metal Tn-seq is in the compendium; Royet is none of its four source studies.
Royet 2025 is loaded on its own as `EnvMetalTnseqRoyet2025Dataset`
([[torchcell.datasets.pputida.royet2025]]).

## 2026.10.10 - Rand 2017 is a source study: sets 1 and 5 are its two days, by content

The `set5` hypothesis ("Rand 2017 ... levulinic acid") and the `set1` hypothesis ("unattributed
(person 'Kelly', Putida_ML5, 2020)") are replaced by a measurement. Rand 2017 (doi
10.1038/s41564-017-0028-z) is not in the literature mirror, so its PMC author manuscript was
deposited in `$DATA_ROOT/torchcell-raw/randMetabolicPathwayCatabolizing2017/` and its
Supplementary Table 1 (LA/Gluc and 4HV/Gluc, 59 genes each) compared against every sample of
this release. set5 LA 40 mM minus set5 glucose reproduces LA/Gluc (r 0.99952, median abs diff
0.041; runner-up treatment 0.264) and set1 4HV 40 mM minus set1 40 mM glucose reproduces
4HV/Gluc (r 0.99984, median 0.025; runner-up 1.959). Full table and nulls:
[[experiments.036-dataset-fixes-before-kg-build.scripts.rand2017_release_inventory]].

`attribute_sample` now returns a `rand2017` source study (basis `source_study_quote`, evidence
`rand_conditions`, `rand_days`, `rand_library`, quotes audited against `torchcell-raw` rather
than `torchcell-library`) for exactly ten samples, keyed by (set, condition, dose) in
`RAND2017_CONDITIONS`: set1 glucose 40 mM, 4HV 40 mM, potassium acetate 20 mM; set5 glucose
and LA 40 mM. Two carry notes: the acetate pair is attributed by the Methods quote only (the SI
prints no acetate value), and the set5 glucose pair is 20 mM in this release where Rand 2017's
Methods state a 40 mM control (the record keeps 20 mM). The remaining set1 and set5 samples (20
mM glucose, 5 mM acetate, 20 mM LA, vanillin, p-coumaric acid) stay compendium-attributed with
rewritten hypotheses.

Coverage after the change (real release): `borchert2024` 111 samples (was 121), 79 kept
(unchanged); `rand2017` 10 samples, 0 kept. **No served record changes**: all ten are on
`RCH2_defined_noCarbon`, which `medium_not_in_media_library` still drops (20 samples, 94,640
records). The medium's recipe is Price 2018 Table S18; what blocks it is compound-identity
rows for its salts (issue #860). When it lands, the 47,320 Rand records carry Rand 2017's
publication.

## 2026.10.11 - The 20 RCH2 samples are kept (#860); Rand 2017's 47,320 records reach the store

`RCH2_defined_noCarbon` now maps to `MEDIA_LIBRARY["RCH2_DEFINED_NOCARBON_PRICE2018"]`,
the recipe Price 2018 Supplementary Table 18 states for the Fitness Browser
([[torchcell.datamodels.media]], 2026.10.11), and the `medium_not_in_media_library` drop
rule is gone. The salts it needed (potassium chloride, sodium dihydrogen phosphate
monohydrate, PIPES sesquisodium salt) and the RCH2 samples' one unresolved carbon source
(`4-Hydroxyvalerate`; `Potassium acetate` was a Price 2018 label) came from the #726
curator pass ([[torchcell.datamodels.compound_identity]], 2026.10.11), which also gave this
store's other 63 gap labels their rows. Before the medium was added, the 20 samples were
checked against the standard-error identity floor `|f / t| >= 0.1` the build re-checks on
every kept Fitness Browser value: 0 violations.

### Measured from the rebuilt store

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/borchert2024_rch2_records.py`
(streams every record, resolves each interned environment), result
`results/borchert2024_rch2_records.json`. Before: the store array 3733 built at main
`52d376d6d` (2026-10-10), read from its `preprocess/dropped_records.json` and
`source_studies.json`. After: slurm 3836 at this branch.

| quantity | before | after |
|---|---|---|
| kept samples | 290 | 310 |
| records (LMDB entries) | 1,372,280 | **1,466,920** (= 1,372,280 + 94,640) |
| records on the RCH2 medium | 0 | 94,640 (20 screens) |
| dropped by `medium_not_in_media_library` | 20 samples, 94,640 records | rule removed |
| dropped by `reactor_process_not_representable` | 22 samples, 104,104 records | unchanged |
| `rand2017` samples / kept | 10 / 0 | 10 / 10 |
| records carrying Rand 2017's publication (doi 10.1038/s41564-017-0028-z) | 0 | **47,320** (set1IT078/079/082/083/084/085, set5IT075/076/081/082) |
| `borchert2024` (compendium-attributed) samples / kept | 111 / 79 | 111 / 89 |

Records by medium after the rebuild: MOPS minimal no-carbon 511,056; the Fitness Browser
glucose no-nitrogen MOPS 492,128; Borchert 2023's modified M9 369,096; RCH2 94,640.
The Rand 2017 attribution is untouched by the medium change: the ten samples keep the
`rand2017` source study and its notes (the acetate pair attributed by the Methods quote
only; the set5 glucose pair at the release's 20 mM against the Methods' 40 mM control).

### L0-L4 after the rebuild

Written by the rebuild itself (slurm 3836, `build_dataset_lmdb --verify` through
`torchcell.datasets.pputida.borchert2024.verify_build`) at the branch's pre-rebase commit
`cb310e339`. It describes the store as built at that commit; every dev store is stale again
on main after later schema changes.

#### `RbTnseqBorchert2024Dataset`

1,466,920 records; store built 2026-10-11 07:13 UTC; 1 report(s); verdict **PASS**.

Report `verification_report.json` (written 2026-10-11 11:28 UTC):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 1466920 records validated |
| L1 | `count` | pass | observed 1466920, expected 1466920 |
| L1 | `pair_uniqueness` | pass | 1466920 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 1466920 values checked |
| L2 | `se_nonnegative` | pass | 953091 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | pass | numeric rule: reference response == 0 for all 1466920 records |
| L3 | `environment_perturbed` | pass | all 1466920 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=30.0, media='MOPS minimal medium, no carbon source (Neidhardt 1974 formulation as tabulated by Price 2018)') |
| L1 | `provenance_gaps` | pass | 7952533 documented provenance gaps over 1466920/1466920 records; 0 deferred field(s): []; 43487080 undeclared None values over 18 carrier fields (top: Compound.inchi x21109452, Compound.chebi_id x4883424, Environment.dilution_rate_per_hour x1466920, EnvironmentResponsePhenotype.category x1466920, EnvironmentResponsePhenotype.category_label x1466920) |
| L1 | `canonical_gene_names` | pass | 4732 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 1944852 compound references carry a structure identifier; 104104 declare a typed gap (2 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 18009992 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 605696 records on a shared MEDIA_LIBRARY medium, 861224 on a medium deriving from one (4 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 4732 measured genes are reference genes (>= 0.9) |
| L4 | `current_genome_genes` | pass | every one of the 4732 measured systematic names is a gene of the current genome |

### After the rebase onto main

The store's `Potassium acetate` compound references (6 across its interned environments,
`results/compound_curator_726_860_store_reach_after.json`) carry the spelling the row had at
`cb310e339`. Main's De Chiara 2022 media spell the same compound `potassium acetate`, so
after the rebase the row's canonical spelling is lowercase
([[torchcell.datamodels.compound_identity]], 2026.10.11) and the next build renames these
references, case only; the InChIKey (`SCVFZCLFOSHCOH-UHFFFAOYSA-M`) and CID (517044) do
not change.

## 2026.10.12 - Thompson 2020's correction read: it rescopes 0 of the 46 samples (#986)

The open item of the 2026.10.07 "Advisory check" is closed by measurement, and nothing
in this loader changed. The owner deposited the correction (PMC8091123; the bytes are
ASM's PDF of DOI 10.1128/AEM.00177-21) under
`$DATA_ROOT/torchcell-raw/thompsonFattyAcidAlcohol2020/si/thompson2021_correction_PMC8091123.pdf`
(sha256 `2c7e387f...`, `manual_browser`). Its one substantive paragraph, verbatim from
the layout text (line breaks folded; the text layer prints "ﬁrst"):

> Readers should note that when the authors created the deletion mutant of PP_2675,
> they deleted the entire region between the gene’s start and stop codons. However, this
> also deleted the DNA that codes for the first 14 amino acids of the overlapping ORF
> PP_2676.

and it replaces the paper's JBEI registry link (folder 456) with folder 641. It touches
the targeted `ΔPP_2675` strain of Thompson 2020's Figs. 7 and 9, and a URL.

Measured against this release (`borchert2024.attribute_sample` over the metadata
sheet, `manual_deposits_2026_10_12.py --measure`): Thompson 2020 is the source of 46
samples (set15 24, set16 12, set12 10), 217,672 records at 4,732 loci, all mutant
library `Putida_ML5_JBEI`, all experiment group `carbon source`, and none names
PP_2675, PP_2676, pedF, a deletion or a knockout. They are pooled RB-TnSeq assays of the
library in wild-type KT2440, so the correction rescopes **0 samples and 0 values** here.
Whether to record the correction on these records anyway is the owner's decision (#986);
this note gives the facts only. Details in
[[experiments.036-dataset-fixes-before-kg-build.scripts.manual_deposits_2026_10_12]].

## 2026.10.12 - Thompson 2019 lysine attributed its four samples (#908)

Owner decision relayed 2026-10-12 (#908, option 1a): the four compendium samples Thompson 2019 lysine measured carry that study as their `SourceStudy`. Branch `feat/fitness-browser-decomposition` (PR #1049).

| sample | Table S1 column | max abs difference (39 loci) | nearest wrong sample |
|---|---|---|---|
| set7IT062 | D-Lysine | 0.0460 | set7IT056 (2.14) |
| set7IT055 | L-Lysine | 0.0194 | set12IT023 (3.22) |
| set7IT044 | 5-AVA | 0.0207 | set7IT057 (1.43) |
| set6IT057 | Glucose | 0.0082 | set8IT058 (0.58) |

Measured by `thompsonMassivelyParallelFitness2019_release_inventory.py` ([[experiments.036-dataset-fixes-before-kg-build.scripts.thompsonMassivelyParallelFitness2019_release_inventory]]).

- `SOURCE_STUDIES["thompson2019_lysine"]`: doi `10.1128/mBio.02577-18` (the raw mirror manifest's casing, the same DOI the Fitness Browser Putida loader writes for `Thompson18`), key `thompsonMassivelyParallelFitness2019`, rank 54 (`build_bacteria_candidate_datasets_table.ranked()` today).
- `attribute_sample` gives the four the new basis `source_release_content_match`, with the Methods' four conditions ("diluted 1:50 into 10 ml MOPS minimal medium supplemented with 10 mM glucose or 5AVA or d-lysine or l-lysine", with the source's U+2009 thin spaces) and Table S1's header (`gene,gene,desc,D-Lysine,L-Lysine,5-AVA,Glucose`) as evidence, each a `SourcedValue` on the raw mirror's pinned files (paper text sha256 `aedb5fb0...`, Table S1 sha256 `b4718373...`). 4 x 4,732 = 18,928 records change publication.
- The Fitness Browser's `pubId` names three more set7 samples `Thompson18` (set7IT042, set7IT056, set7IT057). Table S1 matches none of them, so they stay with the compendium attribution here; the browser loader attributes all seven by `pubId`.
- **set6IT057's vessel: not stored, and nothing stored contradicts the SI.** `Environment` has no vessel field for any bacterial record (vessel and shaking are dropped for every sample, as the 2026.10.07 section says), so neither the paper's "Cells were grown in 50-ml culture tubes" nor the release's "48 well microplate; Tecan Infinite F200" is on a record. The paper's sentence is pinned as `SOURCED_VALUES["thompson_lysine_vessel"]`. Storing a vessel needs `BacterialEnvironmentResponseExperiment.environment` narrowed to `CultureEnvironment` (whose `culture_format.vessel` is verbatim), the route the chemogenomic classes take; that re-fingerprints every served bacterial environment-response closure, so it is a schema-change record and an owner decision, not part of this change.

## 2026.10.13 - Compendium-derived: hidden assumptions and missed representation

This store is derived from Borchert 2024 fModules (`borchertMachineLearningAnalysis2024`). Data collected from a compendium, a database aggregate or a follow-up study may embed that source's preprocessing (normalization, filtering, re-keying, merged conditions) and may lack ontological representation the primary release carries (per-sample metadata, replicate structure, the released statistic). Its records are a derived copy, and the primary release is still to be collected from source ([[torchcell.knowledge_graphs.dataset-admission-loop]], 2026.10.13).

What it may hide here: 254 of the 332 samples are Fitness Browser values copied at three-decimal rounding, so precision is the compendium's; genes without fitness in all 332 samples were eliminated (832 of 5,564 protein-coding genes), a filter of the compendium, not of any source study; the sample grouping and condition labels are the compendium's metadata, not each study's. The Fitness Browser loader now serves the attributed studies from the deposit at full precision; the remaining source studies' own releases are still to be collected.
