---
id: 14qfgon4kolz0di3uqf9wak
title: Pooled_sort_seq
desc: ''
updated: 1791808960177
created: 1791808960177
---

## 2026.10.12 - The pooled sort-seq gate (#991)

`torchcell/verification/pooled_sort_seq.py`, the L0-L4 family gate for
`PooledSortSeqExpressionPhenotype` datasets. It exists rather than reusing
`torchcell/verification/promoter_activity.py` for the same reason the phenotype leaf does:
the arrayed family's L1 rule is one record per (`well_id`, read time), and a pooled library
has no well. The rule that stands in its place is one record per `sequence_id`.

| level | name | what it holds |
|---|---|---|
| L0 | `structural` | every record's experiment validates against `ExperimentType` |
| L1 | `count` | the record count against the registry's oracle |
| L1 | `sequence_uniqueness` | one record per `sequence_id`, the pooled identity rule |
| L2 | `value_fidelity` | finite, inside the bin scale the registry states |
| L2 | `released_values` | every stored estimate equals the released file's own value |
| L3 | `units_declared` | units, reporter and a POOLED readout member on every record |
| L3 | `sequence_is_the_genotype` | the genotype carries this record's sequence, unambiguous ACGT, span inside it |
| L3 | `uncertainty_is_typed_or_gapped` | an absent dispersion carries the typed gap that says so |
| L4 | `read_count_coverage` | how many records carry the release's read count (reported, not required) |

Two of those are the ones a reviewer should look at first. `sequence_is_the_genotype` is
what makes the family a sequence-level genotype store rather than numbers keyed by a
label: it re-checks, per record, that the single promoter-bearing perturbation's
`promoter_variant_id` is the phenotype's `sequence_id`. `units_declared` refuses a
per-well or per-cell `ReporterReadout` member on a pooled record, since such a number
would claim a sample unit the estimate does not have.

The runner is `POOLED_SORT_SEQ_DATASETS` + `verify_pooled_sort_seq_dataset` +
`run_pooled_sort_seq` in `torchcell/verification/runners.py`, called last in `run_all`.
The verifier streams a factory of records, so a store larger than memory is never
materialized; the registry entry states the bin-scale maximum (18.0 for Vaishnav 2022,
the sort-bin count, not the data's range) and the provenance the report carries.

Tests: `tests/torchcell/verification/test_pooled_sort_seq.py` holds each level against
hand-built records; the end-to-end run over a built store is in
`tests/torchcell/datasets/scerevisiae/test_vaishnav2022.py`.
