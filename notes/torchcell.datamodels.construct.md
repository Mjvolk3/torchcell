---
id: 875qhh2v5ms9pw3ncv90ack
title: Construct
desc: ''
updated: 1792054844837
created: 1792054844837
---

## 2026.10.15 - ConstructPlacement, HostPlacement and the contig-relative variant call

`torchcell/datamodels/construct.py` holds the models lane G composes onto
`GeneAdditionPerturbation`. Nothing in `schema.py`'s served classes references them, so
no served closure's fingerprint moves and nothing rebuilds for this module alone. They
are built to be composed, which is the point: a `plasmid_contig_id: str | None` beside a
`locus_tag: str | None` can be half-filled and was `None` in 26 of 26 served
gene-addition stores, while a `placement` is a whole pointer or a typed gap.

### The three objects

| class | the fact it carries |
|---|---|
| `ConstructPlacement` | WHICH DNA: `contig_sha256` + `contig_ref` (both pinning one hash), `feature_key`, `variants`, `host_placement` |
| `HostPlacement` | WHERE in the host: `assembly_set`, `site_label` verbatim, `span` (gap-able), `insert_span` (gap-able), `replaced_span` |
| `ConstructVariantCall` | WHAT differs from the construct's own sequence: a 1-based coordinate on the contig, plus `variant_statement` verbatim |

`Localization` (`chromosomal_integration | episomal_replicon`) lives here too and is
used by no served class yet. Two members, because the replicon class (2 micron, CEN/ARS,
a broad-host-range origin) is a property of the CONTIG, read from its
origin-of-replication feature, not of the gene riding on it.

### Decisions worth stating

- **One insert span, not two.** Review 2 finding 8 asks for `insert_span` on
  `HostPlacement`; the lane spec also lists it on `ConstructPlacement`. Two fields
  holding one fact in two classes is what principle 5 forbids, so the field lives on
  `HostPlacement` (where the integration is described) and
  `ConstructPlacement.insert_span` is a read-only property reading it. An episomal
  construct has none, because the whole replicon is present.
- **Variants are typed, the statement is kept.** `ConstructVariantCall` mirrors
  `BacterialVariantCall`'s field shape and REUSES its enums (`BacterialVariantType`,
  `VariantCallMode`, `VariantFrequencyBasis`) rather than copying them: the same
  physical statement gets the same closed vocabulary. The one difference is the point of
  the class, `reference_contig_sha256` is a hash rather than a free replicon string.
  Cachera's `ARO4-K229L` becomes a coordinate with `variant_statement="K229L"` beside
  it, so the verbatim claim survives and a disagreement stays visible.
- **`call_mode` is optional here, required there.** A DESIGNED allele has no frequency
  to be a fraction of; a frequency without a `call_mode` is refused, because a clone's
  within-isolate fraction and a population allele frequency are different facts.
- **Everything hashes.** `Genotype.__eq__` compares `set(self.perturbations)` and a
  frozen pydantic model hashes the tuple of its field values, so an unhashable nested
  object breaks genotype equality. All three derive from `HashableProvenanceGapMixin`
  (canonical-JSON hash), which is also what lets them carry `provenance_gaps`.
- **A primer cannot be a target.** The validator refuses a `feature_key` whose SO id is
  in `PLACEMENT_TARGET_REFUSED_SO`. 297 of 570 features in the Cachera set are primers.
- **`replaced_span` is a real `None`.** "Replaced nothing" is a fact about the
  integration; "we do not know the coordinate" is a fact about the source. Only the
  second is a gap, which is why `span` and `insert_span` are gap-able and
  `replaced_span` is not.
