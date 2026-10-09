---
id: gp14c5s9wg9de5tldloj9td
title: Environment_response
desc: ''
updated: 1789257764635
created: 1789257764635
---

## 2026.09.12 - Shared rules for the chemogenomic block

The six reviews of the 14 unserved datasets each found the same class of problem: a dataset
reports PASS while carrying something a model cannot use. Vanacloig was "fully sourced" with
name-only compounds, Mota's 0.993 gene containment hid seven records keyed to retired ORFs,
Auesukaree's L3 `reference_zero` passed over zero values because every record is categorical.
The rules below close those, and because none of them is specific to an environment-response
readout they live in a new module, `torchcell/verification/common.py`, as
`SharedRecordRules`: an accumulator that takes one stored record at a time (`add`) and
renders its `LevelResult`s at the end (`results`), so the eager and the streaming verifiers
run the same code and emit the same messages.

### The rules, by level

| Level | Name | What fails it |
|---|---|---|
| L1 | `provenance_gaps` | nothing (informational census), but it now reaches every carrier |
| L1 | `canonical_gene_names` | one systematic name with two common-name spellings; a stored systematic name that is not the genome's current name; a common name that resolves to another gene |
| L2 | `uncertainty_sanity` | a `sample_sd` or `variance` uncertainty of exactly 0 |
| L3 | `compound_identity` | an environment-edit compound with no structure identifier and no typed gap |
| L3 | `media_compound_identity` | the same, for a `defined` component of the base medium |
| L3 | `media_membership` | a medium that is neither a `MEDIA_LIBRARY` member nor derived from one |
| L4 | `gene_containment_sgd` | the aggregate containment floor (unchanged, 0.90) |
| L4 | `current_genome_genes` | any systematic name absent from the current genome's gene set |

**The gap census reaches every carrier.** It used to read `phenotype.provenance_gaps` plus
`environment.provenance_gaps`, which is why a build whose compounds were all name-only could
print "fully sourced". The walk is now structural: any mapping inside the record that carries
a `provenance_gaps` key IS a carrier, which is exactly what `ProvenanceGapMixin` adds, so the
compounds inside perturbations, inside solvents and inside media components are all reached,
and a model that gains the mixin later is reached without editing the traversal. Each carrier
is matched back to its class by its key set (the model field sets, read from
`ProvenanceGapMixin.__subclasses__()`), so the census reports `Compound.inchikey` rather than
a path. Alongside the declared gaps it counts, per `Class.field`, the values that are `None`
with NO gap: the silent Nones. A documented gap stays a PASS and a silent None does not fail
either, but the message no longer says "fully sourced" when there are any.

**Compound identity is split in two results, deliberately.** The rule is one rule, but a
dataset owns the compounds it doses and the shared library owns the components of the medium.
The shipped `MEDIA_LIBRARY` currently has 110 name-only single-substance compounds (`SC` 31,
`SC_URA` 30+1 dropout, the YNB vitamins, YPD's dextrose, agar, the SGA selection agents),
which is a known shared-layer defect already gated by the strict xfail
`test_shared_media_compounds_are_identified_or_gapped` in
`tests/torchcell/datamodels/test_ontology_coherence.py`. Reporting both halves in one line
would bury a chemogenomic set's own unencodable compounds under D-glucose. Only the `YP*`
media are compliant today, which is why the unit-test fixtures use `YP_GALACTOSE`.

**The categorical reference rule.** `_l3_reference_zero` kept its name and its numeric rule
(reference response == 0), and gained a second rule for a dataset whose references carry no
number: every reference carries a category, they are all the SAME category (the declared
baseline), and no experiment record reports that category as a measured call. The result's
`details["rule"]` says which ran (`numeric_zero` or `categorical_baseline`) and the message
starts with it, so a reader can tell a real pass from a vacuous one.

**`screen_id` joins both identity keys.** `_study_key` (publication, `units`, `screen_id`) and
`_condition_signature` (perturbations, temperature, media, durations, `screen_id`) both read
it with `.get`, so a dataset that does not carry the field keys exactly as before. Hoepfner
has 45 same-compound, same-dose column pairs from different screens, each normalized within
its own screen; they are independent measurements, not duplicates.

**L4 is now two results.** The 0.90 containment floor stays for the aggregate, and a separate
`current_genome_genes` fails outright on any systematic name the genome does not carry, with
the per-gene record counts. A floor cannot catch seven records out of 1,273.

### Where the rules run

- `verify_environment_response_dataset` and `verify_environment_response_dataset_streaming`
  both take `resolve_gene_name`, `sgd_genes` and `min_containment` (optional, so the verifier
  still runs with no genome mounted); the streaming path dropped its own gap census and its
  own L4 block in favor of the shared ones.
- `verify_fitness_dataset` takes the same three and runs the shared rules; `run_fitness` no
  longer adds its own containment result (the shared L4 is the same check under the same
  name).
- `verify_segregant_growth_streaming` runs the shared rules WITHOUT `sgd_genes`: a segregant
  genotype is a haplotype mosaic with no gene perturbations, so its gene set comes from the
  genome and its own L4 stays. `canonical_gene_names` reports "no gene perturbations to
  check".
- `run_environment_response` and `run_fitness` in `runners.py` build the S288C genome once
  (`_genome`, `overwrite=False`) and thread `genome.resolve_gene_name` in. No dataset entry's
  `expected_count` was touched; those belong to the per-dataset work.

### Streaming cost

The shared rules are accumulators for the same reason the streaming verifier is: nothing is
materialized. The per-environment verdicts (compound identity, media membership) are memoized
on the IDENTITY of the interned environment mapping, holding a reference to the key object so
the `id()` cannot be reused, which turns a per-record cost into a per-condition one (610
conditions for Hoepfner, not 3.1M records). The gap walk is per record by construction, since
its counts are per record.

### Measured

`pytest tests/torchcell/verification tests/torchcell/knowledge_graphs -x -q` 127 passed; mypy
clean on the seven edited source files; ruff check and ruff format clean. Read-only smoke run
of the eager verifier over the built dev LMDB
`$DATA_ROOT/data/torchcell/env_chemgen_auesukaree2009` (525 records, no genome, no report
written): the new rules fire exactly on what the review predicted, `compound_identity` FAIL
(1-propanol x125, ethanol x95, methanol x55, sodium chloride x42, hydrogen peroxide x30),
`media_membership` FAIL (`YPD (base_medium=None)` x525), `reference_zero` PASS via the
categorical rule (baseline `tolerant`, reported by no experiment record), and the census
reporting 0 gap-capable carriers because that LMDB predates the mixin on `Compound` and
`Environment` (the same staleness its L0 already fails on).

## 2026.10.01 - Eager and streaming verifiers made identical (issue #529)

The streaming verifier's docstring promised it was "semantically identical" to the eager one, and three rows disagreed:

- `pair_uniqueness`: eager counted duplicated KEYS (`len(dups)`) and said "(study, strain, condition) triples appear in multiple records"; streaming counted redundant RECORDS and said "(strain, condition)" although it keyed on the study too. Three copies of one record read 1 against 2.
- `value_fidelity` and `se_nonnegative`: eager ran `l2_value_fidelity` over the list of present values, so a bad value was indexed by its position among them and carried a `reason`; streaming indexed by record with no `reason`.
- `measurement_type_consistent` printed the enum member's repr (`<MeasurementType.log2_ratio: 'log2_ratio'>`), because records hold `model_dump()` output.

Fix: each of these rows is now built by one helper both entry points call (`_pair_key`, `_pair_uniqueness_result`, `_value_problem`, `_value_result`, `_response_value`, `_se_value`, `_measurement_type_result`).

- One duplicate definition, REDUNDANT RECORDS: every record after the first with a key. Three copies are 2. Chosen because it is the unit the `count` row is stated in (`n_pairs + n_duplicated` equals the observed record count), it is the number of records a loader must aggregate or drop for L1 to pass, and a single streaming pass computes it without a per-key counter. Wording: `N records duplicate an earlier (study, strain, condition) triple; M unique triples`.
- Bad values are indexed by RECORD and carry `reason` (`nan`, `inf`, `< 0.0`), so an entry points at the record to inspect.
- Measurement types are collected as `str(...)`, the StrEnum VALUE (`'log2_ratio'`), in both message and details.

No consumer parses these messages: `scripts/verify_datasets.py` and `torchcell/provenance/build_manifest.py` read only `passed` per level. The 13 stored `verification_report.json` files under `$DATA_ROOT/data/torchcell/*/preprocess/` all pass these rows with no duplicates and no bad values, so a regeneration changes only text: every `measurement_type_consistent` message (repr to value), and the `pair_uniqueness` wording of the five streaming datasets (hoepfner2014, wildenhain2015, crispr_magic_lian2019, hillenmeyer2008_het, hillenmeyer2008_hom) from "(strain, condition)" to "(study, strain, condition)". Not regenerated.

Evidence: `test_eager_and_streaming_reports_are_equal` (one release failing every own rule, compared field by field and as whole models), `test_triplicate_record_counts_two_redundant_records`, `test_non_finite_responses_are_indexed_by_record`, `test_negative_se_fails_and_nan_se_is_not_counted`, `test_mixed_measurement_types_fail` in [[tests.torchcell.verification.test_environment_response]].

## 2026.10.01 - Stricter streaming SE rule (review of PR #589)

One behavior change the parity fix introduced and the first write-up did not state: streaming `se_nonnegative` used to flag only `se < 0`, so an infinite SE passed; it now uses the eager rule (`_value_problem` with `minimum=0.0`) and fails an infinite SE with reason `inf`, as the eager verifier always did. NaN SEs are still treated as not reported in both. Measured by the independent review of PR #589 (not re-run here): 0 infinite or negative SEs in the four streaming stores that carry SEs (wildenhain2015, crispr_magic_lian2019, hillenmeyer2008_het, hillenmeyer2008_hom), so no stored `passed` flag would flip on regeneration.

## 2026.10.09 - Three declared reliefs for an ABSOLUTE readout (#776)

Every one is an ORACLE the caller states, not a waiver: the rule passes only while the
observed number equals the declared one, so a regression in either direction fails the
gate. Nothing is relaxed for a dataset that did not ask, and the one relief that is not a
count refuses the request outright for the wrong kind of readout.

### L3 `reference_zero` gains an absolute branch

`verify_environment_response_dataset(..., reference_centered=True)` keeps today's
behavior. `reference_centered=False` runs `_l3_reference_absolute` instead: the
reference's response must be present, finite, and on the SAME `measurement_type` as the
experiment record it references. This is the shape
`torchcell.verification.metabolite`'s `reference_finite` branch already uses for absolute
quantities (Mulleder amino-acid concentrations), and the result row names which of the
three branches ran, as the categorical branch already did.

**The gate is what makes it not a blanket relaxation.** The absolute branch counts every
record whose `measurement_type` is not in `ABSOLUTE_MEASUREMENT_TYPES` and FAILS on any,
so asking for it on a log2-ratio or z-score dataset is an error rather than a silent skip
of the zero check. Measured on a synthetic log2-ratio release:
`n_relative_measurement_type = 3`, rule fails.

### L3 `environment_perturbed` takes a declared unperturbed count

`expected_unperturbed` (default 0) and the rule becomes observed == declared. For a
RESPONSE dataset an unperturbed record is a defect and 0 is right. For an ABSOLUTE readout
the base condition is itself a measured condition: Caglar 2017's base condition was run in
three separate experiments, so 9 of its 55 records carry no environmental edit. Declaring
the count rather than waiving the rule is what keeps those rows from silently growing or
disappearing. Same shape as Bloom 2019's `conditions_documented`, which requires the
no-edit columns to be exactly the absolute-readout columns.

### New L2 `interval_orientation`

Counts stored two-sided intervals that do not bracket their value, against
`expected_non_bracketing` (default 0). A confidence limit carried through a nonlinear
transform can land on the wrong side of the estimate: Caglar 2017 Table S5 releases
`95p = -1027.769034` against a doubling time of 80.95212424, the image of a slope interval
straddling zero under `DT = log_e 2 / slope`. The schema stores the released bytes rather
than repairing them, so this rule is the measurement of how many such rows a dataset
holds. `expected=0` makes any inverted interval a failure, which is the ordinary case;
a dataset that declares a nonzero count passes only while the count holds exactly.

### L1 `_study_key` joins `replicate_id`

A source that releases one row PER REPLICATE CURVE, each with its own interval and fit
quality, has measured that many things. Caglar 2017 Table S5 is 55 rows over 19
conditions; without the replicate id they collapse to 16 unique triples and the rule would
demand a condition mean the paper never released. Measured both ways on a synthetic
three-replicate release: with ids, `n_pairs=3, n_duplicated=0`; without,
`n_pairs=1, n_duplicated=2`. `.get` with a `""` default keeps every other dataset's key
unchanged.

### Both entry points stay identical

The eager and streaming verifiers now carry 17 rows in the same order (the nine own rows,
then the eight shared ones), and `test_eager_and_streaming_reports_are_equal` compares
them field by field. `interval_orientation` and the absolute-reference branch are
accumulators for that reason, so the streaming report is not a second implementation.

Related: [[torchcell.datasets.ecoli.caglar2017_doubling_time]],
[[torchcell.datamodels.schema]].
