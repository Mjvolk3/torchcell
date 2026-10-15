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

## 2026.10.15 - ConstructSpan holds the parts of a compound location

Lane T. The span was one `start`/`end` pair, which cannot hold a GenBank feature whose
location is several intervals, and that cost 11 of the 17 deposited Cachera maps.

### The shape

| field | what it carries |
|---|---|
| `parts: list[SpanPart]` | ordered 1-based end-inclusive intervals, in the order the SOURCE's location string writes them |
| `composition: SpanComposition` | which claim the parts make: `simple`, `origin_spanning`, `joined` |
| `orientation` | the strand of the whole feature; one feature reads in one direction |

`SpanPart` is two coordinates and nothing else, forward-strand, because the orientation
is a property of the feature.

### Why `origin_spanning` is not `joined`

They are different claims about the same number of intervals, and the vocabulary is the
INSDC feature table's, not an invented one. INSDC `join` means "The indicated elements
should be joined (placed end-to-end) to form one contiguous sequence". A feature that
crosses the origin of a CIRCULAR replicon is **one** interval whose coordinates wrapped
and whose parts are adjacent in the molecule; a two-part **join** is two intervals with
sequence between them that the feature does not cover. Collapsing them would make a
666 bp `pUC` origin and a spliced two-exon gene the same shape.

The distinction cannot be derived from the parts: it needs the replicon's length and
topology, which the span does not carry. So the claim is STORED and
`ConstructContig._check_span` proves it against the bytes (circular topology, first part
ending at the last base, every part in range). The topology itself is grounded in SO's
`topology_attribute` SO:0000986 through `SO:0000988 circular`.

INSDC `order` is a THIRD claim ("nothing is implied about the reasonableness about
joining them") and is not a member: 0 of 570 measured features use it, so a
representation would be built against nothing. It is refused loudly with its own typed
reason, as are a fuzzy bound, a between-bases site, a mixed-strand join, a remote
accession reference, a backwards join, a wrap on a linear replicon and an unparsable
location.

### `start` and `end` raise for a compound span

They are properties now, and they serve the `simple` case only. A compound span has no
single `start..end`, and the thing that would be returned is the envelope, which is the
exact defect this change removes: `pBTX002.gb`'s `pUC\ori` as the whole 12,096 bp
plasmid. A consumer written for one interval fails loudly, which is the same discipline
that made the parser refuse what it could not hold. `length` is the sum of the parts and
is correct for all three compositions.

### The round trip is the proof

`ConstructSpan.insdc_location()` writes the canonical INSDC location the parts mean, and
`insdc_location(compact_wrap=True)` writes the `11559..128` form the mirrored maps carry.
Both are generated from the same parts, so no field records which spelling the source
used: the syntax is not a fact about the molecule. 570 of 570 features of all 17 maps
regenerate their own file's location string exactly, and every compound span's canonical
spelling re-parses through Biopython into the same parts.

Two INSDC spellings are NOT regenerated, and the docstring says so: a single base is
written `7..7` rather than `7`, and a reverse-strand join as
`complement(join(a..b,c..d))` rather than the equivalent
`join(complement(c..d),complement(a..b))`. Nothing in these 17 maps uses either, and a
span cannot tell them apart from the form it does write, which is why the round trip is
checked per deposited set rather than assumed.

### Cachera's `insert_span` after this change

Still a typed gap, and that is the measured answer, not a deferral. The four records
point at pBTX1, which carries no NotI site and annotates only a 211 bp up-arm fragment,
so the arms admit 9,850 bp and 9,377 bp against the paper's "9.3 kb". pBTX2, newly
indexed, DOES admit an exact interval -- 1529..10798 = 9,270 bp, from the paper's "NotI
excised insert of pBTX2" and the two `GCGGCCGC` sites at 1527 and 10797 -- but that is
pBTX2's coordinate space and no record points at it. Full measurement in
[[experiments.036-dataset-fixes-before-kg-build.scripts.cachera2023_construct_span_parts]].

Record: `database/schema_changes/2026-10-15-construct-span-holds-the-parts-of-a-compound-location.json`,
`rebuild_scope: full_kg_rebuild`, 3 served datasets. The breaking removal of `start` and
`end` rests on the standing Path U authorization; the span shape has not been reviewed by
the owner and is put forward with its measurements.
