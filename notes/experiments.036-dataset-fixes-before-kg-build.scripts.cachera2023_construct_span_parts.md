---
id: otgukzavrehgnjc3il0h3b1
title: Cachera2023_construct_span_parts
desc: ''
updated: 1792086728056
created: 1792086728056
---

## 2026.10.15 - What a span of parts buys the Cachera set, and what it does not

`experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_construct_span_parts.py`
re-measures the deposited `cachera2023_cri_spa_plasmids` index now that `ConstructSpan`
holds ordered parts, and settles the one question the unblocking was supposed to answer.
Results:
`experiments/036-dataset-fixes-before-kg-build/results/cachera2023_construct_span_parts.json`.

### Coverage

| | before (span = one interval) | after (span = parts) |
|---|---|---|
| files listed | 17 | 17 |
| contigs indexed | 6 | **17** |
| blocked maps | 11 | **0** |
| features held | 208 | **570** |
| `simple` spans | 208 | 559 |
| `origin_spanning` spans | 0 | **11** |
| `joined` spans | 0 | 0 |
| names quoted from `paper.md` | 6 | 8 |
| names quoted from SI Table S2 | 0 | **9** |
| names typed absent | 0 | 0 |
| `provenance_complete` | True | True |

The before row is the deposited index of 2026-10-15T08:51, read before it was trashed
(`/scratch/projects/deprecated/2026.10.15/122139-manifest.json`).

### The round trip

570 of 570 features of all 17 maps regenerate the location string their own file wrote,
byte for byte, from the stored parts and strand alone
(`ConstructSpan.insdc_location(compact_wrap=True)` against the raw FEATURES block, read
without Biopython because Biopython is what rewrites the compact wrap form). 11 of 11
compound spans also pass the other direction: the canonical
`join(11559..12096,1..128)` spelling re-parses through Biopython into the same parts.

### The 11 features that were blocking, by name

Every one is an interval across the origin of a circular replicon, written in the compact
`start..end` form with `start > end`. None of the 17 files contains a `join` or an
`order` operator anywhere in its FEATURES block; the "11 join locations" the audit
counted were Biopython's rewrite of these same 11 wraps, so origin-wrap and join were one
finding, not two.

| map | feature | source location | parts (1-based) | feature bp | plasmid bp |
|---|---|---|---|---|---|
| `pBTX002.gb` (pBTX2) | `rep_origin pUC\ori` | `11559..128` | 11559-12096, 1-128 | **666** | 12,096 |
| `pCCM023.gb` | `CDS AmpR` | `complement(2997..858)` | 2997-2999, 1-858 | **861** | 2,999 |
| `pbtx013.gb` | `CDS AmpR` | `complement(4892..858)` | 4892-4894, 1-858 | **861** | 4,894 |
| `pbtx014.gb` | `CDS AmpR` | `complement(4889..858)` | 4889-4891, 1-858 | **861** | 4,891 |
| `pbtx016.gb` | `CDS AmpR` | `complement(4890..858)` | 4890-4892, 1-858 | **861** | 4,892 |
| `pbtx017.gb` | `CDS AmpR` | `complement(4893..858)` | 4893-4895, 1-858 | **861** | 4,895 |
| `pbtx019.gb` | `CDS AmpR` | `complement(4890..858)` | 4890-4892, 1-858 | **861** | 4,892 |
| `pbtx020.gb` | `CDS AmpR` | `complement(4892..858)` | 4892-4894, 1-858 | **861** | 4,894 |
| `pbtx021.gb` | `CDS AmpR` | `complement(4898..858)` | 4898-4900, 1-858 | **861** | 4,900 |
| `pbtx022.gb` | `CDS AmpR` | `complement(4890..858)` | 4890-4892, 1-858 | **861** | 4,892 |
| `pbtx023.gb` | `CDS AmpR` | `complement(4890..858)` | 4890-4892, 1-858 | **861** | 4,892 |

The last column is what the envelope stored before the parser was hardened: a 666 bp
origin of replication read as a 12,096 bp replicon, and a 861 bp beta-lactamase (286
residues and a stop, which is what `bla` is) read as a whole plasmid.

### Cachera's contig-side insert span: exact on pBTX2, still a gap on pBTX1

Four `GeneAdditionPerturbation` records carry the placement, and all four point at
**pBTX1** (`pBTX001(ante113).gb`, handle `bbaca60509e0`), because the paper says CD-Btx
was built with pBTX1 and the CRI-SPA screen transfers that cassette into the YKO library.
Measured on both vectors:

| | pBTX1 (CD-Btx, the records' contig) | pBTX2 (BY-Btx) |
|---|---|---|
| length | 15,098 bp | 12,096 bp |
| up arm | `XII-5 up B` 10..220 (211 bp) | `XII-5\UP` 1549..2002 (454 bp) |
| down arm | `XII-5 dw A` 9598..9859 (262 bp) | `XII-5\Down` 10107..10782 (676 bp) |
| arm to arm, inclusive | 10..9859 = **9,850 bp** | 1549..10782 = **9,234 bp** |
| between the arms | 221..9597 = **9,377 bp** | 2003..10106 = **8,104 bp** |
| NotI sites (`GCGGCCGC`) | **none** | 1527 and 10797 |
| NotI fragment (top-strand cuts) | not defined | 1529..10798 = **9,270 bp** |
| the paper's own figure | "the considerable size, 9.3 kb, of the Btx-cassette" | same sentence |

On **pBTX2 the question has an exact answer**: the paper states "The BY-Btx strain was
made by transforming the NotI excised insert of pBTX2", the bytes carry exactly two NotI
sites, and `GC^GGCCGC` cuts bound 1529..10798 = 9,270 bp, which IS the paper's 9.3 kb.
That interval is now recorded here and is the `insert_span` of the round-trip example in
`tests/torchcell/datamodels/test_datamodels_roundtrip.py`. pBTX2's own quoted name
statement in the construct index is that NotI sentence, so the store carries the evidence.

On **pBTX1 it does not**. pBTX1 has no NotI site, its up arm is annotated only as a 211 bp
`XII-5 up B` butted against the map's origin at position 10, and the two readings the arms
admit (9,850 and 9,377 bp) bracket the paper's 9.3 kb without matching it. A
double-crossover's transferred interval is bounded by crossover points inside the arms,
which the source never states, so there is no exact pBTX1 interval even in principle.
`HostPlacement.insert_span` therefore stays a `ProvenanceGap` and the gap note now names
the pBTX2 contrast. Moving pBTX2's number onto a pBTX1 placement would be inventing a
coordinate, which is the one thing this lane was told not to do.

The HOST-side `span` likewise stays a gap: the paper names the pad "XII-5" and publishes
no chromosome XII coordinate, and the only XII-5 coordinate in the release
(`target sequence XII-5` at 14691..14710 on pBTX1) is a coordinate on the vector.

The 2026.07.06 section of
[[torchcell.sequence.plasmid-and-genomic-content-design]] records the arms as
`XII-5\UP` 1548-2002 and `XII-5\Down` 10106-10782 and the insert as 8,104 bp. The arm
coordinates are off by one against the bytes (1549 and 10107); the 8,104 bp figure is
right for the between-the-arms reading, 2003..10106.

### The placement census

`construct_placement_census` over the four Btx-cassette perturbations the loader builds:
4 additions, 0 `sequence_backed`, **4 `host_placed_without_coordinate`**, 0
`composition_only`, 0 `n_untyped`, all on contig `bbaca60509e0`. Unchanged by this lane,
and correctly so: the row reads `sequence_backed` only when the host site's `span` is
set, and that coordinate is not published. What moved is the construct set behind it, 6
contigs to 17.
