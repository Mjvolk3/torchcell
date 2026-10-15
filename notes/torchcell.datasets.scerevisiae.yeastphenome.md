---
id: czf1blmi8sxbwq7yjor01eq
title: Yeastphenome
desc: ''
updated: 1790814227090
created: 1790814227090
---

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against per-screen `valuez_sha256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.12 - Every released record stored as released; no genome at load time (#1045)

Owner rule, 2026-10-12: the data is the data; the genome and the experiment are loaded separately and meet through their gene sets at modeling time, so a loader never consults the annotation to decide whether a record exists or how it is keyed. Sweep row: [[database.genome-gating-sweep.2026.10.12]]. PR #1079 (group 1). The store is rebuilt from the branch with `--retire-existing --verify` (slurm array 4457); the L0-L4 table is appended when the task finishes.

Before: a column was stored only when its zygosity was haploid or homozygous diploid, its phenotype a growth readout, its medium a base in the two library tables, and its condition one dosed agent in `_UNIT_MAP`; every other growth column was counted and left out: 53 condition columns (237,558 values), 14 media columns (63,859), 2 heterozygous columns (6,862).

Now (`_parse_conditions`, `_parse_media`): a condition string is a comma-joined list; `standard` is the base condition with no edit; `temperature [N C]` sets the environment temperature (no temperature gap then); `time [N h]` / `[N gen]` set the durations; `oxygen [0%]` sets `aerobicity="anaerobic"`; each dosed agent is a `SmallMoleculePerturbation` (defensins stay `BiologicPerturbation`); `%`, `% v/v`, `% w/v` join `_UNIT_MAP`; a repeated identical component counts once. A `het` column is an `EngineeredCopyNumberPerturbation` (2 -> 1, KanMX) in BY4743. A medium base outside the two tables is classified by prefix (`YP` rich; `SC`, `SD`, `YNB`, `CSM`, `MM`, `SynBase` synthetic), anything else stores `is_synthetic` as a typed gap, and a 4-field header stores a `Media` whose name, state and synthetic flag are typed gaps (the SGD precedent). 133 environments, 524,682 records (was 83 and 296,777).

Not stored, counted in `preprocess/dropped_records.json` under `condition_not_representable`: 19 columns (80,374 released values) whose condition has a component no schema field carries: a dose RANGE (`hydrogen peroxide [1-1.2 mM]` x7 in berry_gasch_2011, `caspofungin [10-30 ng/ml]`, `[60-120 ng/ml]`), an agent with no dose (`gentamicin`, `Actara`, `Mospilan`, `Confidor`, `16001701`; `Concentration` needs a value or a basis), a unit `ConcentrationUnit` lacks (`ng/ml`: `rapamycin [4 ng/ml]`; `U/ml`: `zymolyase [30 U/ml]`), and a genotype edit written as a condition (`loss + cytoduction of mtDNA`, `expression of human CYP1A2`, `expression of human NAT2`). The two units are an additive enum change for the schema PR; the rest need a carrier. One `expression of mtDNA (ARG8m)` column (4,473 values) is a reporter readout, not growth, and stays out of this growth loader. A row id with no ORF form is counted under `deferred_to_released_gene_name_leaf` (0 today).

## 2026.10.12 - The deferred name is a record on the released-gene-name leaf (#1045, PR #1085)

A row id with no ORF form is stored on `ReleasedGeneNamePerturbation`, kind `heterozygous_deletion` for a `het` column and `deletion` otherwise, marker KanMX (0 of the released rows); the ledger key is `stored_on_released_gene_name_leaf`. The count stays 524,682.

### Family verification (job 4606) and what it took to pass

The first family-verifier pass over the rebuilt store (slurm 4606) failed three gates, each a property of the released matrices rather than a defect: `pair_uniqueness` found 14,508 records repeating an earlier (study, strain, condition) triple (the curated matrices release the same strain under one parsed condition in more than one column of a screen; the repeated records are identical in units, medium and edit, measured over all 524,682 records), `environment_perturbed` found 8,975 records with no environmental edit (the base-condition columns), and `media_membership` found every medium free text because `_parse_media` set no `base_medium`. The loader now anchors each released medium token at the `MEDIA_LIBRARY` key its leading word or classifying prefix names (`YPD`, `SC`, `SD`, `YNB`, `YP` for `YPG [3%]`, `LB`), keeping the released name as `name` and substituting no library object; the runner spec declares the 14,508 repeated pairs (`expected_duplicated_pairs`) and the 8,975 unperturbed records (`expected_unperturbed`), both read by the verifier as the loader's own count, so an undeclared repeat still fails. Store rebuilt by slurm 4657 and verified by 4658; the L0-L4 table follows.

### L0-L4 from the rebuilt store (slurm 4673, family verifier job 4674)

**yeastphenome**: verdict PASS (/scratch/projects/torchcell-scratch/data/torchcell/yeastphenome/preprocess/verification_report.json)

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 524682 records validated |
| L1 | `count` | gate | ok | observed 524682, expected 524682 |
| L1 | `pair_uniqueness` | gate | ok | 14508 records duplicate an earlier (study, strain, condition) triple (14508 declared by the loader); 510174 unique triples |
| L2 | `value_fidelity` | gate | ok | 524682 values checked |
| L2 | `se_nonnegative` | gate | ok | 0 values checked |
| L2 | `interval_orientation` | gate | ok | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | gate | ok | single measurement_type: 'z_score' |
| L3 | `reference_zero` | gate | ok | numeric rule: reference response == 0 for all 524682 records |
| L3 | `environment_perturbed` | gate | ok | 8975 of 524682 experiments carry no environmental edit, as declared (a measured condition with no edit: an absolute readout's base condition or a released control arm; baseline temp=None, media='YPD') |
| L1 | `provenance_gaps` | gate | ok | 2247227 documented provenance gaps over 524682/524682 records; 1 deferred field(s): ['inchikey']; 10319349 undeclared None values over 22 carrier fields (top: Environment.dilution_rate_per_hour x524682, Environment.duration_generations x524682, EnvironmentResponsePhenotype.assay_type x524682, EnvironmentResponsePhenotype.category x524682, EnvironmentResponsePhenotype.category_label x524682) |
| L1 | `canonical_gene_names` | report | ok | 6068 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | gate | ok | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | gate | ok | environment edits: 228574 compound references carry a structure identifier; 195918 declare a typed gap (47 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | gate | ok | medium components: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | gate | ok | 0 records on a shared MEDIA_LIBRARY medium, 524682 on a medium deriving from one (13 distinct media) |
| L4 | `gene_containment_sgd` | report | ok | 1.000 of 6068 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | report | ok | every one of the 6068 measured systematic names is a gene of the current genome |

## 2026.10.13 - Compendium-derived: hidden assumptions and missed representation

This store is derived from YeastPhenome yp-data (`turcoGlobalAnalysisYeast2023`). Data collected from a compendium, a database aggregate or a follow-up study may embed that source's preprocessing (normalization, filtering, re-keying, merged conditions) and may lack ontological representation the primary release carries (per-sample metadata, replicate structure, the released statistic). Its records are a derived copy, and the primary release is still to be collected from source ([[torchcell.knowledge_graphs.dataset-admission-loop]], 2026.10.13).

What it may hide here, measured against two primary releases: yp-data negates Parsons 2006's released log2 ratio ([[experiments.database.scripts.parsons2006_yeastphenome_crosscheck]]), and averages Yoshikawa 2009's two cultures per condition while pooling strains its ORF translation merges into one locus ([[experiments.database.scripts.yoshikawa2009_yeastphenome_crosscheck]]). Every screen is rescaled to a mode-referenced value, and fields the curation does not carry are typed gaps. Each screen's own release is the source to collect.

## 2026.10.15 - 9,671 of the 14,508 declared duplicate pairs are an aerobic and an anaerobic record

Correction to the 2026.10.13 statement above that the 14,508 repeated pairs "are identical in units, medium and edit". They are identical in the GENOTYPE (no field of any stored perturbation differs in any of the 14,508 groups), but **9,671 of them differ in `environment.aerobicity`**: one ORF, one screen, one medium, an aerobic record and an anaerobic one. The condition signature the family verifier keyed on read seven hand-picked environment fields and `aerobicity` was not among them, although `ENVIRONMENT_IDENTITY_FIELDS` -- what decides whether two records share one environment node in the served graph -- does read it.

With the signature reading every field the served environment identity reads, this store's L1 `pair_uniqueness` becomes **519,845 unique triples, 4,837 duplicated**, and the runner spec declares 4,837. The 4,837 that remain differ only in the released response value and are the two-columns-of-one-screen case the loader describes. Nothing about the stored records changes.

A second finding on the same store, not yet fixed: the dropout, carbon-source and buffer media state their composition only in `Media.name`. `SC - Ade`, `- Arg`, `- Lys`, `- Met`, `- Trp` (4,195 records each), `SC (-C)`, `SC, Glycerol [2%]` and `SC, HEPES [25 mM]` all carry `components: []` and `dropouts: []`, so the served environment identity cannot tell them from plain `SC` (212,402 records) and puts six of them on one environment node. One fact, one field puts the dropout in `Media.dropouts` and the carbon source and the buffer in `Media.components`, sourced from the release; that changes stored records, so it needs its own lane with a rebuild of this store.

Measured by `experiments/database/scripts/environment_response_identity_outcomes.py`; reasoning and the whole-family table in [[torchcell.verification.environment-response-identity]].

## 2026.10.15 - The released medium token is stored whole, and its composition is typed

The second finding of the section above is fixed, and verifying it found that the residual
4,837 declared duplicate pairs were the SAME defect rather than repeats.

`_parse_media` stored `media_field.split("+")[0]` as `Media.name`, which had two consequences.
The `+` half of the token reached no field at all, so **474,837 of 524,682 records carried a
name that is not the released token**: six tokens stored the plain name `SC` (212,402 records,
which is lane W's figure exactly), six stored `YPD` (239,081) and five stored `SD` (23,354).
And what the retained half states sat inside a name `media_identity` drops, so the 27 stored
tokens reached **8** media identities and **3 served environment nodes were reached from more
than one released medium, over 53,951 records**: the five SC dropout arms with their complete
control (30,628), two vehicle controls with a buffered medium (13,649), and Galardini 2019's
nitrogen-free arm with its own control (9,674).

That last pair is the whole residual L1 allowance. `SD + His + Leu + Lys + Ura, -N` and
`SD + His + Leu + Lys + Ura (2)` are two columns of one screen, both condition `standard`,
and both stored the bare name `SD`, so the **4,837 pairs this store declared were a
nitrogen-starvation arm counted as a repeat of its control**. Aggregating them, which is what
a declared allowance invites, would have destroyed 4,837 measurements.

`Media.name` is now the token VERBATIM and `_parse_media_token` reads what it states into
typed fields: a named dropout into `dropouts` through the shared resolver (`SC - Ade` ->
adenine); a carbon source, a buffer, a vehicle, a selection agent or an auxotroph supplement
into `components` at its stated dose; and a nutrient CLASS the curation declares absent without
naming the compound into the new `Media.dropout_roles` (`SC (-C)` -> `[carbon_source]`,
`SD ..., -N` -> `[nitrogen_source]`). Every medium carries one `SourcedValue` quoting the token
against the sha256 of the screen file whose header states it. An unknown token form RAISES.
`YPG [3%]` keeps its 3 % against an unexpanded `composition_deferred` carbon-source line,
because the curation abbreviates that ingredient to a letter and never writes it out. The base
recipe stays unexpanded, as before: no library object is substituted for a token.

After: **27 distinct stored names, 28 media identities, 125 environment identities, 0 records
on an environment node reached from a second released medium, and 0 L1 duplicated records**
(524,682 unique triples). `expected_duplicated_pairs` is **0**, from 4,837.
`gene_essentiality_sgd`'s `YPD with 2% glucose` carried the same shape and is fixed in the
same change. Reasoning, the per-token table, the cross-dataset scan and the growth-phase
measurement: [[experiments.database.scripts.media_name_composition_scan]].
