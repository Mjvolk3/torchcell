---
id: 1qmme0jzucbzuk4ve94gv6l
title: Construct_placement
desc: ''
updated: 1792054852957
created: 1792054852957
---

## 2026.10.15 - Counting the three typed states of a gene addition's material

`torchcell/verification/construct_placement.py` counts audit B 2.5's three states over a
stream of perturbation mappings (the JSON dicts a built LMDB yields), as
`provenance_gap_census` does. Importable and tested; wiring it into the family runners
waits for lane G to put `placement` on the addition leaf.

| state | what it means |
|---|---|
| `sequence_backed` | `placement` set, bytes pinned by an `ArtifactRef` |
| `host_placed_without_coordinate` | `placement` set with a host site whose `span` is itself gapped: a named landing pad, no published coordinate |
| `composition_only` | `placement` absent with a typed `ProvenanceGap`; the parts are stored exactly as they are today |

`n_untyped` is counted apart from all three: a `placement` of `None` with no gap naming
it is the silent `None` the typed states exist to abolish, and folding it into
`composition_only` would report a defect of the record as a property of the source.

The row is **report-class**, registered by name in `report.GENOME_REFERENCE_RULES`
(`construct_placement_states`, added 2026-10-15, with
`tests/torchcell/verification/test_finding_class.py`'s pinned registry updated to
match). Its subject is the record against OUR sequence references, so it carries its
statistic under the verdict and never fails a store: a source that released no plasmid
map is a correct store of that source. What the count buys is a worklist that names
exactly what a retrieval campaign should fetch. Audit B section 4 measured the honest
starting number: **1 of 15 datasets fillable from the mirror today** (Cachera), 3 more
plausible through JBEI handles, 11 gapped, 1 (Mohiuddin) not even mirrored.

`gap_reasons` tallies WHY, which is what makes the worklist actionable, and is the
consumer of `ProvenanceGapReason.not_released_by_primary` (added in the same change):
Kang 2026's sequences are supplied "upon request" and Carruthers 2025's are "verified by
whole plasmid sequencing" and not deposited, so neither is `not_reported_by_primary`
(the primary did report it) nor `deferred_pending_source_review` (no artifact of ours
holds it).
