---
id: hz3dic31osj4992u79r9xi3
title: Verdict
desc: ''
updated: 1791617762255
created: 1791617762255
---

## 2026.10.10 - The verdict contract

`CandidateVerdict` is frozen with `extra="forbid"` and composes one `GateResult` per gate, exactly G1 to G5 in order. It is the contract the `/add-dataset` agents (piece 3) receive as `verdict_json_schema()` and return as JSON that `candidate-gate validate` checks with `model_validate_json`. Named for the candidate on purpose: `AdmissionReport` already means serving admission.

Validators, each tested in `tests/torchcell/candidates/test_verdict.py`:

- `GateResult`: a `gap` names its issue; `blocked_kind` is set for, and only for, `blocked`. Blocked kinds: `wall`, `provision`, `ingest`, `unfiled_gap`, `deposit`.
- Stop rule: after the first `fail` or `blocked`, every later gate is `unmeasured`. A `gap` does not stop the run.
- G1 and `source_kind` agree: `pass` with `primary` or `aggregation`, `fail` with `transcription` or `prediction`, `unmeasured` with `None`. An `AggregationRecord` is present exactly when the kind is `aggregation` (owner decision 2026-10-10, plan decision 15).
- A measured gate carries its sub-record (`g2`, `g3`, `g4_key`, `g5`) and an unmeasured one carries none; `g4_value` needs `g4_key`. The `G2Record` outcome must equal what its hosts imply (the worst host decides) and must map to the gate's outcome (`G2_GATE`).
- `torchcell_commit` is 40 hex characters; `decided_at` starts with an ISO date.

`outcome` is a derived property, never stored, so a stored verdict cannot disagree with its gates: `refused` on the first fail, `blocked:<kind>` on the first blocked, `pending` when any gate is unmeasured, `admissible_with_gaps` when the only non-passes are filed gaps, else `admissible`. `passing` is the last two, which is what the enforcement test requires.

`AggregationRecord` fields: `n_source_studies`, `attribution_field`, `value_origin` (`re_measured` or `derived_by_aggregator`), `n_sources_mirrored` and `net_new_vs_served` (`None` means not measured, never zero), and quoted `evidence`. Evidence everywhere is the existing `SourcedValue` (file + sha256 + verbatim quote), deliberately outside the fingerprinted schema surface.

Pinned to pydantic 2.12.3 semantics: nothing depends on `PydanticUserError` being a `RuntimeError` or on discriminated-union serialization falling back, the two behaviors 2.13.0 changes.

## 2026.10.11 - Per-source decomposition (plan decision 16)

`AggregationRecord.sources: tuple[SourceDecomposition, ...] | None`, additive and optional (`None` until measured). One entry per (source study, served class). Fields: `source_label` (the compendium's attribution value), `role` (`original_study`, `compendium_paper`, `unattributed_release`), `citation_key` and `mirrored`, `doi`, `dataset_class`, `n_samples_in_compendium`, `n_samples_served`, `own_release` with `own_release_kind` (`per_record_values`, `subset_values`, `raw_reads_only`, `none`), `own_release_url` and `own_release_access` (`scriptable`, `walled`), `loaded_as_own_dataset` with `own_dataset_class`, `cross_check: SourceCrossCheck | None`, `note`.

Rules: the compendium's own paper has no separate release; `own_release` agrees with its kind; URL and access exist exactly when there is an own release; `mirrored` exactly when a key is named; a loaded study names its class; served never exceeds compendium samples; a cross-check needs an own release. On the record: a (label, class) pair appears once, entries sharing a label agree on role, key and DOI, and `n_source_studies` and `n_sources_mirrored` equal the decomposition's counts. `SourceCrossCheck.passes` reads a difference as passing at or below its tolerance and a correlation at or above; no tolerance means a pin (`None`). Properties `loadable_from_own_release` and `compendium_only` drive the counts. Policy and measurements: [[torchcell.candidates.aggregation-decomposition]].
