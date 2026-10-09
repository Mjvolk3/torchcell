---
id: qvh8d829smyuv3sav8b618h
title: Schmidt2016_s23_growth_rate
desc: ''
updated: 1791555321435
created: 1791555321435
---

## 2026.10.09 - Table S23 loaded as 15 absolute growth rates (#826, E. coli audit rank 13)

Rank 13 of [[plan.bacteria-si-phenotype-audit-ecoli]] was blocked for three stated
reasons. Re-measured after PR #836 (#776) only the third survives, and it is not a
blocker. The loader is `torchcell/datasets/ecoli/schmidt2016_s23_growth_rate.py`, class
`GrowthRateS23Schmidt2016Dataset`, store `data/torchcell/growth_rate_s23_schmidt2016`.

| the audit's blocker | re-measured 2026.10.09 |
|---|---|
| no absolute growth member, so L3 `reference_zero` is unsatisfiable | LIFTED: `MeasurementType.growth_rate` is in `ABSOLUTE_MEASUREMENT_TYPES`, so `reference_centered=False` is admissible and the reference states its own finite rate |
| two rows are negative, which a log2 ratio cannot hold | MOOT on the absolute route: `EnvironmentResponsePhenotype` never clamps, and both negative rows drop for an unrelated reason |
| the `Stdev`'s replicate design is not back-solvable | CONFIRMED, and handled with a conservative `n_samples` rather than by dropping the statistic |

### The pinned bytes

`si2.xlsx`, sha256 `3280a13ff67a73f25440cff6ee73fb99b5ce3ef57854213dbbf6272be241912f`,
the same workbook `schmidt2016.py` consumes from the same raw mirror under the same pin;
`paper.md` sha256 `67bedae8f934421086c23b7fa9582b0950a07206bffe7c4ffdd11e4d003a8710`.
Nothing is retrieved that `schmidt2016.py` does not already record.

### 26 released rows, 15 records, and the 11 drops

`Table S23` releases 12 columns (columns 10 and 11 are the unlabelled continuation cells
of `OD @ harvesting. replicates`) over 26 rows: BW25113 22, `MG1665` 2, `NCM3722` 2.
No cell of the 26 x 12 block is blank; what reads blank is typed text, `-` in the two
stationary rows and `after >6 volume changes` in the four chemostat rows.

| rule | rows | why |
|---|---|---|
| `strain_not_distinguishable_from_bw25113` | 4 | `NCM3722` is in neither `BacterialReferenceStrain` nor the deposited assembly sets. `MG1665` is a transposition of `MG1655`, and correcting it does not rescue it: every row is wild type, so L1 `pair_uniqueness`'s genotype signature is the empty tuple for all three strains and its condition signature carries no strain field. Measured on 17 BW25113 + typo-corrected MG1655 records: "2 records duplicate an earlier (study, strain, condition) triple; 15 unique triples" |
| `medium_has_no_media_library_entry` | 1 | `Glycerol + AA` is M9 + glycerol + 21 named supplements, which no `MEDIA_LIBRARY` key states |
| `growth_phase_not_representable` | 2 | `Environment` has no growth-phase slot, so both stationary rows collapse onto the Glucose environment |
| `culture_not_batch` | 4 | `Environment` has no culture-mode or dilution-rate slot, so the four chemostat arms collapse onto one environment |

The last two rules are the proteome loader's own (`schmidt2016.CONDITIONS`), imported
rather than restated, and the collapse they describe was re-measured on RECORDS: a
19-record build with the chemostat rows in fails L1 with "3 records duplicate an earlier
(study, strain, condition) triple; 16 unique triples" and L2 `uncertainty_sanity` with
"4/19 labeled uncertainties are a sample dispersion of exactly 0" (the chemostat `Stdev`
is literally `0`, the dilution-rate set-point).

### The 15 records

LB 1.9, Acetate 0.3, Fumarate 0.42, Galactose 0.26, Glucose 0.58, Glucosamine 0.46,
Glycerol 0.47, Pyruvate 0.4, Succinate 0.44, Fructose 0.65, Mannose 0.47, Xylose 0.55,
Osmotic-stress glucose 0.55, 42 C glucose 0.66, pH6 glucose 0.63 (h^-1). The reference
is the released Glucose row itself, 0.58 h^-1, which is ALSO a record: for an absolute
readout a reference condition is a measured condition too, as Caglar 2017's base
condition is. 14 of the 15 carry an `EnvironmentPhysicalPerturbation` or a
`SmallMoleculePerturbation`; LB carries none, because its edit is its complex medium,
which L3 `environment_perturbed` counts as an edit (non-baseline media).

### The `Stdev`, and the one inferential step

The paper never states what the released `Stdev` is computed over, and a back-solve over
all 30 sheets of the pinned workbook is NEGATIVE:

- `Table S24` is a different cultivation. Its glucose wild-type `Average` is 0.5921 with
  `Stdev` 0.01202 against this table's 0.58 +/- 0.01; acetate 0.31033 +/- 0.00566 against
  0.30 +/- 0.04. Neither the means nor the SDs match.
- `Table S28` has `Replicate 1/2/3` headers, but its triplicate columns are protein MASS
  ("Standard deviation between triplicates" against `Relative cytosolic protein mass`);
  its `Experimental Growth rate` column restates this table's single value.
- Table S23's own three unlabelled replicate cells are `OD @ harvesting`, optical
  densities rather than rates.
- The released `Stdev` values are rounded to 1-2 significant figures (`0.003`, `0.004`,
  `0.005`, `0.01`), so no identity test on them is even determinate.

Two candidate denominators are released and neither is tied to the statistic: the
triplicate cultivation, stated verbatim as "We grew E. coli BW25113 (ref. 19) under 22
different growth conditions in biological triplicates." and corroborated by Table S25's
three sample files per condition and by this table's own three OD-at-harvesting cells;
and the Methods' "The growth rate was calculated from at least four consecutive
measurements", which is a fit WITHIN one curve. `n_samples = 3` is the conservative
resolution of that pair, the lower count and so the larger standard error, and it is the
only one with a sourced number. `SOURCED_VALUES["n_samples"]`'s note carries the
inferential step into the record. Measured both ways: the 15-record build passes all 16
L0-L4 rows with `n_samples = 3`, and also passes with the `Stdev` left unstored behind
two typed gaps.

### `assay_type`

`AssayType.other`, not a typed gap: the paper DOES state the assay ("the numbers of
cells taken for LC-MS/MS analyses were determined for each sample by flow cytometry"),
so a gap would claim ignorance the release does not create. `liquid_od_growth` would be a
false type, because the released OD is the harvest point and not the readout the rate is
fit to. `SOURCED_VALUES["assay_type"]` carries the quote and the reason.

### What is not stored

Eight released columns have no axis in the schema and are written verbatim to
`preprocess/not_stored.json` with every value rather than coerced onto one:
`Single cell volume [fl]1`, `Doubling time (h-1)`, `Time exp before harvest (h)`,
`# of doublings at exponential growth before harvesting`, the three
`OD @ harvesting. replicates` cells and `Number of Proteins Identified (FDR 1%)2`.

Recorded, not acted on: Table S28 restates the two stationary-phase conditions'
`Experimental Growth rate` as `0` where Table S23 releases `-0.01`, an internal
inconsistency of the release on two rows this loader drops for an unrelated reason.

### Why a separate module

`build_manifest` keys a built store's staleness on the schema closure of the loader
MODULE's own `torchcell.datamodels` imports (`provenance/schema_deps.py:loader_closure`),
so importing `EnvironmentResponsePhenotype` into `schmidt2016.py` or
`schmidt2016_growth_rate.py` would mark the served `proteome_schmidt2016` or
`growth_rate_schmidt2016` store stale for a change that touches none of their records.
The pinned artifact, the media objects, the condition table and the environment helpers
are imported FROM `schmidt2016`, so there is one copy of each.

### L0 to L4, measured on the built dev store

`PYTHONPATH=<wt> python -m torchcell.datasets.ecoli.schmidt2016_s23_growth_rate verify`,
`DATA_ROOT=/scratch/projects/torchcell-scratch`. All 17 rows PASS.

| level | rule | result |
|---|---|---|
| L0 | structural | 15 records validated |
| L1 | count | observed 15, expected 15 |
| L1 | pair_uniqueness | 15 unique (study, strain, condition) records, one each |
| L1 | provenance_gaps | 19 documented gaps over 15/15 records; 1 deferred field (`inchikey`); 366 undeclared None values over 14 carrier fields |
| L1 | canonical_gene_names | no gene perturbations to check |
| L2 | value_fidelity | 15 values checked |
| L2 | se_nonnegative | 15 values checked |
| L2 | interval_orientation | 0 of 0 stored intervals do not bracket their value, as declared |
| L2 | uncertainty_sanity | 15 labeled uncertainties, none a zero dispersion |
| L3 | measurement_type_consistent | single measurement_type: `growth_rate` |
| L3 | reference_zero | absolute rule: reference value finite and on the record's own scale for all 15 records |
| L3 | environment_perturbed | all 15 experiments carry an environmental edit (baseline temp 37.0, media the Schmidt M9 base) |
| L3 | compound_identity | 12 compound references carry a structure identifier; 4 declare a typed gap |
| L3 | media_compound_identity | 169 medium-component references carry a structure identifier; 0 gaps |
| L3 | media_membership | 15 records on a shared MEDIA_LIBRARY medium (2 distinct media) |
| L4 | assembly_pin_resolves | all 15 records pin `ecoli_K12_BW25113_ASM75055v1` / `GCA_000750555.1` |

### Schema impact

None. No symbol of `torchcell/datamodels/schema.py` is touched, so no served dataset is
staled by this loader; the new class is additive and reaches the graph through its own
adapter module, conf and `kg_bacteria.yaml` entry.
