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
