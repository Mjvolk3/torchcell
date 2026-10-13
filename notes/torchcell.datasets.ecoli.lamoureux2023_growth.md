---
id: m8gx57g0lduinyetncum0qz
title: Lamoureux2023_growth
desc: ''
updated: 1791556783925
created: 1791556783925
---

## 2026.10.09 - PRECISE-1K's growth-rate column loaded as 89 absolute rates (#826, rank 11)

Rank 11 of [[plan.bacteria-si-phenotype-audit-ecoli]] was blocked "in its absolute form
by gap 1". Gap 1's second half is closed by PR #836 (#776). The loader is
`torchcell/datasets/ecoli/lamoureux2023_growth.py`, class
`GrowthRateLamoureux2023Dataset`, store `data/torchcell/growth_rate_lamoureux2023`: 89
records plus one reference, no schema change.

### The pinned bytes, and the column

`metadata_qc.csv`, sha256
`68a0c5aa13a4fe09245d0da73691794978afbc87784d6fdbcc472829b43481c2`, one of the four
files `lamoureux2023.py` already mirrors from the Zenodo archive. Nothing new is
retrieved. The readout is the released header `Growth Rate (1/hr)`, and that header is
the ONLY statement of the readout and its unit: grepping `paper.md` for "growth rate"
returns ONE hit, "growth rate and oxidative stress during naphthoquinone-based aerobic
respiration", which defines nothing. The paper never names the instrument, the OD range
or the fitting method.

### The counts, each measured

| step | count |
|---|---|
| rows of `metadata_qc.csv` | 1,035 |
| rows carrying a `Growth Rate (1/hr)` | 354 |
| of those, `p1k_*` ids of the PRECISE-1K index | 354 of 354 |
| kept by the expression loader's own genotype and environment rules | 103 |
| of those, exactly `0.0` | 14 |
| stored records | 89 |

The 251 drops are the expression loader's rules, imported and not restated: 138 evolved
isolates, 94 heterologous constructs, 12 non-MG1655 strains and 7 point-mutation alleles.
Measured: the 103 the rules admit are exactly the built `rnaseq_lamoureux2023` records
that carry a rate. The sibling Public K-12 arm contributes NOTHING, because every rate
cell is a `p1k_*` id: 0 of its 240 built records carry one.

### The 14 zero-rate rows are dropped, and that is a decision

A stored 0 h^-1 asserts a fully arrested culture from which a sequencing library was
nonetheless prepared, and the release carries no legend that says so: a 0 in a released
numeric column is indistinguishable from an unrecorded cell, and whether the authors
meant 0 or blank is stated nowhere in the paper or the release. The conservative reading
is that the cell is uninterpretable, so the 14 rows are dropped under
`zero_rate_not_distinguishable_from_unrecorded` with every id, condition and value in
`preprocess/dropped_records.json`, which makes the decision reversible. All 14 sit in 7
stress conditions: `fur:wt_dpd`, `fur:delfur_dpd`, `ompr:wt_nacl`, `oxidative:wt_pq`,
`oxidative:deloxyr_pq`, `oxidative:delsoxr_pq`, `oxidative:delsoxs_pq`.

The cost is recorded rather than hidden: the `oxyR`, `soxR` and `soxS` deletions appear
ONLY in those rows, so they are absent from the stored gene set, which holds 13 deleted
genes (`cra` 6 records, `ydcI` 4, `yheO` 4, and `fur`, `yafC`, `yeiE`, `yiaJ`, `yieP`,
`ybaO`, `ybaQ`, `ybiH`, `yddM`, `pdhR` 2 each; 34 of the 89 records carry a deletion and
55 are wild type). The remaining 89 rates run 0.07 to 1.42 h^-1 and none is negative.

### The reference is a declared base condition, never a borrowed row

The absolute branch needs the reference to state its own finite value on the records'
own scale. The expression loader's reference cannot carry it: measured, both
`control:wt_glc` samples `p1k_00001` and `p1k_00002` release an EMPTY rate cell. So the
reference is the declared base condition, wild-type MG1655 in M9 with `glucose(2)` at
37 C and pH 7.0, and its value is the mean over every stored record whose genotype is
wild type and whose environment serializes identically to that base environment.

| sample | condition | rate (1/hr) |
|---|---|---|
| p1k_00161 | ica:wt_glc | 0.58 |
| p1k_00162 | ica:wt_glc | 0.58 |
| p1k_00163 | ica:wt_glc | 0.66 |
| p1k_00164 | ica:wt_glc | 0.66 |
| p1k_00185 | ica:wt_glc | 0.63 |
| p1k_00186 | ica:wt_glc | 0.63 |
| p1k_00191 | ytf:wt_glc | 0.69 |
| p1k_00192 | ytf:wt_glc | 0.68 |

Mean 0.63875 h^-1, spread 0.11. The ninth `*:wt_glc` row with a rate,
`ssw__wt_glc__1` at 0.73, is NOT in the aggregate: it was grown on `glucose(4)`, a
different environment, and the byte comparison of the serialized environment drops it
without a hand-written exclusion. An aggregate rather than one row because the eight span
19%, which is the Caglar 2017 situation verbatim: "the three released base measurements
span 0.2174 log2 ... so borrowing one is not neutral".

### `screen_id` is the released project, and it is load-bearing

Measured over the 89: the L1 key (genotype, environment, project, `rep_id`) is unique for
all 89, and dropping the project leaves 2 duplicate triples, because `ica:wt_glc` and
`ytf:wt_glc` are the same declared environment and the same wild-type genotype measured
in two different projects, which repeat `rep_id` 1 and 2. This is Caglar 2017's use of
`screen_id`: the source's own run label, not a synthesized encoding of a factor the
schema cannot hold.

### Five typed gaps on every record, and why each one is a gap

| field | why |
|---|---|
| `assay_type` | the release states neither the instrument nor the fit, and the paper's single mention of a growth rate defines nothing. Unlike Schmidt 2016, where the assay IS stated and `AssayType.other` carries it, here `liquid_od_growth` would be a reading rather than a released fact |
| `environment_response_uncertainty` and its type | no released column states a dispersion: measured, the only metadata column whose name matches any of `sd`, `std`, `err`, `ci` or `sem` is `Sequencing Machine` |
| `environment_response_se` | no dispersion is released, so none reduces to a standard error |
| `n_samples` | `Biological Replicates` counts the condition's RNA-seq libraries, not measurements of the rate, and nothing ties them. Measured evidence that a released row is not a culture: within `ica:wt_glc` the six libraries carry 0.58, 0.58, 0.66, 0.66, 0.63 and 0.63, three values on three pairs |
| `sample_unit` | the released row is a sequencing library; the release never says what one measurement of the rate is |

The reference carries two more: `screen_id` (it aggregates two projects) and
`replicate_id` (it aggregates several released replicates).

### Why a separate module, measured

`loader_closure` on `lamoureux2023.py` is 61 symbols and equals the 61 the served
`rnaseq_lamoureux2023` store records; adding the environment-response symbols raises it
to 72, so that store (and nothing else) would read STALE for a change touching none of
its 241 records. The pinned artifacts, the metadata reader, `settle_genotype`,
`settle_environment`, `build_environment`, the media library and the sourced values are
imported FROM `lamoureux2023`.

### L0 to L4, measured on the built dev store

`PYTHONPATH=<wt> python -m torchcell.datasets.ecoli.lamoureux2023_growth verify`,
`DATA_ROOT=/scratch/projects/torchcell-scratch`. All 17 rows PASS.

| level | rule | result |
|---|---|---|
| L0 | structural | 89 records validated |
| L1 | count | observed 89, expected 89 |
| L1 | pair_uniqueness | 89 unique (study, strain, condition) records, one each |
| L1 | provenance_gaps | 691 documented gaps over 89/89 records; 1 deferred field (`inchikey`) |
| L1 | canonical_gene_names | 13 systematic names, one canonical spelling each, each current in the genome |
| L2 | value_fidelity | 89 values checked |
| L2 | se_nonnegative | 0 values checked (no SE is stored) |
| L2 | interval_orientation | 0 of 0 stored intervals do not bracket their value, as declared |
| L2 | uncertainty_sanity | 0 labeled uncertainties, none a zero dispersion |
| L3 | measurement_type_consistent | single measurement_type: `growth_rate` |
| L3 | reference_zero | absolute rule: reference value finite and on the record's own scale for all 89 records |
| L3 | environment_perturbed | all 89 experiments carry an environmental edit |
| L3 | compound_identity | 176 environment-edit compound references carry a structure identifier; 34 declare a typed gap |
| L3 | media_compound_identity | 34 medium-component references declare a typed gap (3 distinct unencodable compounds) |
| L3 | media_membership | 89 records on a medium deriving from a MEDIA_LIBRARY entry (8 distinct media) |
| L4 | gene_containment_mg1655_b_numbers | 13 of 13 deleted loci are MG1655 GenBank gene rows |

### Schema impact

None. No symbol of `torchcell/datamodels/schema.py` is touched, so no served dataset is
staled; the new class is additive and reaches the graph through its own adapter module,
conf and `kg_bacteria.yaml` entry.

## 2026.10.11 - The growth-rate arm's compounds after the #726 curator pass

This arm passes the same PRECISE-1K cells through the resolver as
[[torchcell.datasets.ecoli.lamoureux2023]], so the same rows reach it: rebuilt by slurm array
3809 (task 17), 18 of 18 references to curated labels carry an InChIKey (0 of 24 before),
with the same name changes (`Kanamycin` -> `kanamycin`, the unstereo amino acids -> the
`L-` rows) and no loader code change.

### L0-L4 after the rebuild

#### `GrowthRateLamoureux2023Dataset`

89 records; store built 2026-10-11 07:10 UTC; 1 report(s); verdict **PASS**.

Report `verification_report.json` (written 2026-10-11 07:10 UTC):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 89 records validated |
| L1 | `count` | pass | observed 89, expected 89 |
| L1 | `pair_uniqueness` | pass | 89 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 89 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'growth_rate' |
| L3 | `reference_zero` | pass | absolute rule: reference value finite and on the record's own scale for all 89 records (['growth_rate']) |
| L3 | `environment_perturbed` | pass | all 89 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media='M9 (PRECISE-1K; recipe not stated) + sauer trace element mixture') |
| L1 | `provenance_gaps` | pass | 623 documented provenance gaps over 89/89 records; 0 deferred field(s): []; 2243 undeclared None values over 17 carrier fields (top: Compound.inchi x416, Compound.chebi_id x296, Compound.inchikey x172, Compound.pubchem_cid x172, Compound.smiles x172) |
| L1 | `canonical_gene_names` | pass | 13 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 210 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 34 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 89 on a medium deriving from one (8 distinct media) |
| L4 | `gene_containment_mg1655_b_numbers` | pass | 13 of 13 deleted loci are MG1655 GenBank gene rows |

## 2026.10.12 - #1057 on main: released rows stored, rebuilt and verified

Owner rule of 2026-10-12: a released row is stored under the identifier its release gives, and the genome never gates storage. Rebuilt from main `db632f2b0` with `--retire-existing --verify` (slurm array 4506; the stores with no per-dataset verifier entry point verified by job 4582).

### `growth_rate_lamoureux2023`: 89 -> 260 records

Row 22 and the #1057 dispatch: evolved isolates, BW25113 samples, rpoB alleles and the 16 released 0.0 rates.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 260 records validated |
| L1 | `count` | pass | observed 260, expected 260 |
| L1 | `pair_uniqueness` | pass | 260 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 260 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'growth_rate' |
| L3 | `reference_zero` | pass | absolute rule: reference value finite and on the record's own scale for all 260 records (['growth_rate']) |
| L3 | `environment_perturbed` | pass | all 260 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media='M9 (PRECISE-1K; recipe not stated) + s... |
| L1 | `provenance_gaps` | pass | 1854 documented provenance gaps over 260/260 records; 1 deferred field(s): ['inchikey']; 6694 undeclared None values over 18 carrier fields (top: Compound.inchi x1131, Compound.... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 1 systematic names are not the genome's current name; 1 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 552 compound references carry a structure identifier; 34 declare a typed gap (6 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 51 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 260 on a medium deriving from one (10 distinct media) |
| L4 | `gene_containment_mg1655_b_numbers` | pass | 18 of 18 deleted loci are MG1655 GenBank gene rows |

## 2026.10.12 - #1045 part 2: the rest of the released rows, under their released identifiers

Every row #1057 still dropped that is not an author withdrawal is stored, and no identifier is translated: the stored key is the one the release prints, and a crosswalk to an annotation is a modeling-time join outside the record. Rebuilt from the part-2 branch with `--retire-existing --verify` (slurm array 4583; the stores with no per-dataset verifier entry point verified by jobs 4645 and 4644).

### `growth_rate_lamoureux2023`: 260 -> 354 records

The 94 pColi construct samples, on the expression loader's dispatch.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 354 records validated |
| L1 | `count` | pass | observed 354, expected 354 |
| L1 | `pair_uniqueness` | pass | 354 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 354 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'growth_rate' |
| L3 | `reference_zero` | pass | absolute rule: reference value finite and on the record's own scale for all 354 records (['growth_rate']) |
| L3 | `environment_perturbed` | pass | all 354 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media='M9 (PRECISE-1K; recipe not stated) + s... |
| L1 | `provenance_gaps` | pass | 2700 documented provenance gaps over 354/354 records; 1 deferred field(s): ['inchikey']; 9420 undeclared None values over 18 carrier fields (top: Compound.inchi x1601, Compound.... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 19 systematic names are not the genome's current name; 19 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 740 compound references carry a structure identifier; 128 declare a typed gap (7 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 51 compound references carry a structure identifier; 94 declare a typed gap (1 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 354 on a medium deriving from one (11 distinct media) |

## 2026.10.13 - #1045 part 3: the oxygen, growth-phase and released-strain samples stored

The schema gained a typed gap on Environment.aerobicity, the regime 'transition', Environment.growth_phase and PhysicalFactor.dissolved_oxygen (record 2026-10-13-environment-aerobicity-gap-transition-growth-phase-dissolved-oxygen), so no released sample of these loaders is dropped. Rebuilt from the part-3 branch (`c61b1310a`) with `--retire-existing --verify` (slurm arrays 4640 and 4643; the stores with no per-dataset verifier entry point verified by job 4641).

### `growth_rate_lamoureux2023`: 354 -> 354 records

Rebuilt because the Lamoureux environment builder it shares changed.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 354 records validated |
| L1 | `count` | pass | observed 354, expected 354 |
| L1 | `pair_uniqueness` | pass | 354 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 354 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'growth_rate' |
| L3 | `reference_zero` | pass | absolute rule: reference value finite and on the record's own scale for all 354 records (['growth_rate']) |
| L3 | `environment_perturbed` | pass | all 354 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media='M9 (PRECISE-1K; recipe not stated) + s... |
| L1 | `provenance_gaps` | pass | 2700 documented provenance gaps over 354/354 records; 1 deferred field(s): ['inchikey']; 9420 undeclared None values over 18 carrier fields (top: Compound.inchi x1601, Compound.... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 19 systematic names are not the genome's current name; 19 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 740 compound references carry a structure identifier; 128 declare a typed gap (7 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 51 compound references carry a structure identifier; 94 declare a typed gap (1 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 354 on a medium deriving from one (11 distinct media) |

## 2026.10.13 - Compendium-derived: hidden assumptions and missed representation

This store is derived from PRECISE-1K (`lamoureuxMultiscaleExpressionRegulation2023`). Data collected from a compendium, a database aggregate or a follow-up study may embed that source's preprocessing (normalization, filtering, re-keying, merged conditions) and may lack ontological representation the primary release carries (per-sample metadata, replicate structure, the released statistic). Its records are a derived copy, and the primary release is still to be collected from source ([[torchcell.knowledge_graphs.dataset-admission-loop]], 2026.10.13).

What it may hide here: the `Growth Rate (1/hr)` column is compendium metadata, curated across many studies, not a released measurement table; how each rate was fit, from how many replicates, and what the 16 zero rates mean are the source studies' to state. The source studies' own growth measurements are the source to collect.
