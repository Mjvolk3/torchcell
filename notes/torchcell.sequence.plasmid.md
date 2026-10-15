---
id: re5hs9oed1a8poek9pxx1u5
title: Plasmid
desc: ''
updated: 1783563784291
created: 1783563784291
---

## 2026.07.08 - Plasmids as first-class total-genomic-content

This module exists so a plasmid that is physically PRESENT in a cell during phenotype collection becomes a first-class part of the genotype, not an untracked footnote. Under torchcell's stance that a perturbation is an edit to the TOTAL genomic content in the cell (deletions, integrations, alleles, present plasmids, heterologous cassettes), owning plasmid sequence + annotation as pydantic objects is what makes both sequence-level genotype fidelity and future inverse strain design representable.

- Owns an SBOL/Sequence-Ontology-aligned SUPERSET (`Component`/`Feature`/`Location`/`SORole`), so we interoperate with the standards while capturing more than they mandate.
- Ingests GenBank first (BioPython) with sha256-pinned provenance; GenBank/GFF3/SBOL writers are deliberately deferred future work.
- Present-content vs construction-source is a property of how a perturbation REFERENCES a Component, not of the sequence: a plasmid used only to build a strain becomes a chromosomal `GeneAddition`, not stored cell content.
- `feature_sequence` (flank + circular wrap) and `subcomponent` (design composition) are the extraction/design operations the store is built to serve.

## 2026.10.15 - A feature holds several SBOL Locations, and the refusals are typed

Lane T. `parse_genbank_component` used to refuse an origin-wrapping or compound location
(and before that, to store its envelope). It now HOLDS the three INSDC location forms
this project meets in deposited DNA, and refuses the rest loudly and by name.

### The model

`Location` is unchanged and is still exactly an SBOL Range: one 0-based half-open
interval with an orientation. What changed is `Feature`, which now carries
`locations: list[Location]` plus `composition: SpanComposition` -- SBOL's own shape,
since a SequenceFeature has 0..* Locations. The envelope of several ranges is never a
Location of its own.

- Every range of one feature shares one orientation. That validator IS the refusal of a
  mixed-strand `join`: one feature reads in one direction.
- `Feature.location` is a property serving the single-range case and RAISES for a
  multi-range one, so a caller written for one range cannot silently receive the
  envelope. `Feature.length` is the sum of the ranges.
- `feature_sequence` slices every range and concatenates them in the order the source
  wrote them, which is what INSDC's `join` means, and adds `flank` at the two OUTER ends
  only. For `pBTX002.gb`'s `pUC\ori` that returns 666 bp instead of 12,096.
- `subcomponent` keeps a feature only when EVERY range is inside the window, so a
  feature that crosses the origin is never half-carved.

### Part order is the source's, not Biopython's

Biopython hands a compound location's parts back in FEATURE order (5'->3'), so for a
reverse-strand feature it reverses what the location string wrote: `complement(18..2)` on
a 20 bp circular map parses to `[(0, 2), (17, 20)]`. The parser re-reverses when the
strand is -1, so the stored order is the written order and the location string
regenerates exactly.

### `LocationRefusal`, one vocabulary

`GenBankLocationError` now carries `.refusal`, a `LocationRefusal` member, so the
construct index records the reason the PARSER gave rather than one hardcoded where the
refusal is caught. `registry.BlockedReason` is this same enum, re-exported.

| member | the case, and why it is not representable |
|---|---|
| `ordered_location` | INSDC `order`: "nothing is implied about the reasonableness about joining them", a weaker claim than `join`; 0 of 570 features use it |
| `fuzzy_position` | `<345..500` says the feature extends past the recorded bound; a `SpanPart` is two exact coordinates |
| `between_bases_site` | `123^124` is a site, which Biopython returns as a zero-length interval; a site is not an interval |
| `mixed_strand_location` | `join(complement(1..10),20..30)` is a trans-spliced claim |
| `remote_reference_location` | `J00194.1:100..202` points at another accession's bases |
| `unordered_join` | a `join` whose parts run backwards and is not the origin crossing; reordering it would be a warn-and-fix |
| `origin_wrap_on_linear_topology` | a crossing on a replicon the LOCUS line calls linear: two statements of one file disagree |
| `unparsable_location` | Biopython set the location to `None` (`3.6`, `?`); there is nothing to store |
| `origin_wrapping_location` | RETIRED, now held; kept so the index deposited 2026-10-15T08:51 still loads |
| `compound_location` | RETIRED, same reason (`RETIRED_LOCATION_REFUSALS`) |

A compact wrap on a LINEAR record lands in `unparsable_location`, not
`origin_wrap_on_linear_topology`: Biopython applies its rewrite only to circular records.
