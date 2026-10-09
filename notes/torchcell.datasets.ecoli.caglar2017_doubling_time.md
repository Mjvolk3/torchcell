---
id: 5z16uha8doytlsfmc5kjtmw
title: Caglar2017_doubling_time
desc: ''
updated: 1791534327281
created: 1791534327281
---

## 2026.10.09 - Table S5 loaded: 55 per-replicate doubling times as an absolute readout

Closes the loadable half of [#776](https://github.com/Mjvolk3/torchcell/issues/776).
`DoublingTimeCaglar2017Dataset` serves Caglar 2017 Supplementary Table S5 as 55
`BacterialEnvironmentResponseExperiment` records, one per released biological-replicate
growth curve of wild-type *E. coli* B REL606.

Source: `data/srep45303-s6.csv` in the Caglar raw mirror, sha256
`76411accacbdc28310622cc15289b65ad44937bdc915051bbcfc8c4da1b04c60`, retrieved
2026-10-09 by `torchcell.literature.retrieve.pmc_cloud_object` on key
`PMC5394689.1/srep45303-s6.csv`. The reference row comes from `data/srep45303-s2.csv`
(Table S1), sha256 `1486290bf6a340ae64ee20c915435c0a00ff5eede489de1f56b62733b66f8940`.

### Why a separate module

`build_manifest` keys a built store's staleness on the schema closure of the loader
MODULE's own `torchcell.datamodels` imports, so importing `EnvironmentResponsePhenotype`
into `caglar2017.py` would mark the served `rnaseq_caglar2017` and `proteome_caglar2017`
stores stale for a change that touches none of their records. The pinned artifacts, the
media objects, the environment helpers, the sourced-value helpers and the publication are
imported FROM `caglar2017`, so there is one copy of each. Same reason
`schmidt2016_growth_rate.py` is separate from `schmidt2016.py`.

### What the record stores

| field | value |
|---|---|
| `measurement_type` | `growth_rate` (the enum's own description is "absolute or normalized growth rate / doubling time") |
| `assay_type` | `liquid_od_growth` |
| `environment_response` | the released doubling time in minutes |
| `environment_response_lower` / `_upper` | the released `95m` / `95p`, verbatim |
| `confidence_level` | 0.95 |
| `n_samples` / `sample_unit` | 1 / `biological_replicate` (one curve per record) |
| `replicate_id` | Table S5's own `replicate` index |
| `screen_id` | Table S1's own `experiment` label |
| `units` | `doubling time in minutes (log_e 2 / slope of the linear fit to OD600)` |
| `environment_response_se` | typed `not_reported_by_primary` gap |

### The three declared counts, every one measured on the pinned bytes

Measured by the build (`preprocess/build_accounting.json`) and independently by
`experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_doubling_time_loadability.py`:

| oracle | value | what it is |
|---|---|---|
| `EXPECTED_RECORDS` | 55 | 19 conditions, 3 replicates each except 2 for exactly `Gluconate.tab` and `Lactate.tab`, as the Methods state |
| `EXPECTED_UNPERTURBED` | 9 | the base condition was run in three separate experiments (`Glucose.tab`, `MgSO4_000.800_mM.tab`, `NaCl_005_mM.tab`); for an absolute readout each is a measured condition, not a defect |
| `EXPECTED_NON_BRACKETING` | 1 | `Glycerol.tab` replicate 1, value 80.95212424, `95m` 38.94241445, `95p` **-1027.769034** |

The build raises on any other shape, and the verifier requires observed to EQUAL declared
for the last two, so neither count can drift silently.

### Why the interval is stored as two limits and not repaired

`UncertaintyType.ci95` is a single half-width, and Table S5's interval is asymmetric in
**55 of 55 rows** (upper-to-lower half-width ratio min -26.3920, median 1.3187, max
10.1066; median absolute asymmetry 2.5783 min). Feeding the upper side to `ci95` derives
`environment_response_se = -565.6845`, a negative standard error, which `se_nonnegative`
would then fail. A negative doubling-time upper bound is what the image of a slope
interval straddling zero becomes under `DT = log_e 2 / slope`, so that row has no
half-width at all. Both limits are released bytes and both are stored; the L2
`interval_orientation` rule counts the one that does not bracket its value against the
declared 1.

### Why the reference states 53.679955 min and not 0

The verifier's L3 `reference_zero` numeric branch requires the reference response to be
0, and the base condition's doubling time is 53.68 min; a stored 0 would assert instant
growth. The absolute branch (`reference_centered=False`) requires instead that the
reference carry a finite value on the record's own scale, and it refuses any record whose
`measurement_type` is not in `ABSOLUTE_MEASUREMENT_TYPES`.

The value is Table S1's RELEASED condition-level fit for the `glucose_time_course` base
condition: 53.679955 min, `95m` 49.100298, `95p` 59.201792. Table S5 releases only
per-replicate numbers, and the arithmetic mean of its three glucose replicates is a
number the paper never released (measured: Table S1's value equals none of the
arithmetic, geometric or harmonic means of them). Table S1 is read for that ONE row and
is never loaded as a record, which is what keeps the two fits of one OD600 experiment
from being stored twice.

`n_samples` on the reference is a typed gap: the paper never says what Table S1's fit is
computed over, so asserting 3 would be a guess.

The base condition appears three times in Table S5 (three experiments measured it), with
per-condition means 53.2538, 61.9140 and 58.3515 min, a 0.2174 log2 spread. Which of
them is the reference changes NOT ONE stored number, because the readout is absolute; the
spread is recorded in `preprocess/base_condition_spread.json` as a dataset-level
limitation and is the reason the log2-ratio form was rejected.

### Why not a log2 ratio

Measured, all four candidate forms, in
`experiments/036-dataset-fixes-before-kg-build/results/caglar2017_doubling_time_loadability.json`:
the only form that passed every rule before this change stored **11 of 19** conditions,
because the `MgSO4_stress_low` series releases no 0.8 mM Mg2+ curve of its own, so 5
conditions and 15 of the 55 rows have no in-series reference. A doubling-time ratio is
also a quantity the paper never released.

### What is not stored

`preprocess/not_stored.json`: Table S5's `r.squared` (55 values) and the reference row's
own `rSquared` (1 value). No phenotype class in the schema has a fit-quality axis
(`FluxPhenotype` carries a confidence interval and no r^2), so deriving one would be
inventing a field rather than sourcing it. Tables S6 to S14 are untouched by this module.

### L0 to L4 on the built dev store

`$DATA_ROOT/data/torchcell/doubling_time_caglar2017`, 55 records, PASS, 17 rows:

| level | rule | result |
|---|---|---|
| L0 | `structural` | 55 records validated |
| L1 | `count` | observed 55, expected 55 |
| L1 | `pair_uniqueness` | 55 unique (study, strain, condition) records, one each |
| L1 | `provenance_gaps` | 112 documented gaps over 55/55 records; 1 deferred field (`inchikey`) |
| L1 | `canonical_gene_names` | no gene perturbations to check |
| L2 | `value_fidelity` | 55 values checked |
| L2 | `se_nonnegative` | 0 values checked (the SE is a typed gap) |
| L2 | `interval_orientation` | 1 of 55 stored intervals do not bracket their value, as declared |
| L2 | `uncertainty_sanity` | 0 labeled uncertainties, none a zero dispersion |
| L3 | `measurement_type_consistent` | single measurement_type: `growth_rate` |
| L3 | `reference_zero` | absolute rule: reference value finite and on the record's own scale for all 55 records |
| L3 | `environment_perturbed` | 9 of 55 experiments carry no environmental edit, as declared |
| L3 | `compound_identity` | 44 compound references carry a structure identifier; 2 declare a typed gap |
| L3 | `media_compound_identity` | 213 medium components carry a structure identifier; 0 gaps |
| L3 | `media_membership` | 55 records on a shared `MEDIA_LIBRARY` medium (2 distinct media) |
| L4 | `assembly_pin_resolves` | all 55 records pin `ecoli_B_REL606_ASM1798v1` / `GCA_000017985.1`, the accession the deposited assembly report names |

`l4_assembly_pin` is this dataset's L4 because every record is wild type: no record names
a gene, so the shared L4 gene-containment rules have nothing to look at and are left off
rather than run vacuously against a universe no record names. What the records DO assert
about the outside world is their genome pin, and that is what the rule re-derives.

### Schema impact

`scripts/schema_impact_check.py --base origin/main`: **34 impacted datasets, 0 breaking**,
every change an added optional field. This lands with the KG 4.0 full rebuild, never an
incremental admission.

Related: [[torchcell.datasets.ecoli.caglar2017]],
[[torchcell.verification.environment_response]],
[[plan.bacteria-si-phenotype-audit-ecoli]], [[plan.bacteria-si-phenotype-audit-pputida]].
