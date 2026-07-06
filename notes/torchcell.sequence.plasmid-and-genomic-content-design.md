---
id: r394indwzkzwr8t7mv0oyzn
title: Plasmid and Genomic Content Design
desc: ''
updated: 1783362951372
created: 1783362951372
---

## 2026.07.06 - Total-Genomic-Content Principle + Plasmid Sequence Store

### Principle (signed off 2026-07-06)

What we embed is the **total genomic content present in the cell during phenotype
collection**. A plasmid is captured as PRESENT CONTENT only if it is physically in the
cell while the phenotype is measured. A plasmid used only to CONSTRUCT the strain is not
content -- its retained effect is a chromosomal sequence MANIPULATION, and the
construction plasmid is merely the SOURCE we extract the inserted segment from.

Two cases, keyed by `GeneAddition.localization`
(`[[torchcell.datamodels.gene-addition-perturbation-design]]`):

- **`episomal_*` (present plasmid):** the plasmid is an ADDITIONAL CONTIG of the cell's
  genome. Applying the `GeneAddition` adds that contig to the genomic content.
- **`chromosomal_integration` (construction source):** the integrated segment (between
  the integration-site homology arms) is INSERTED into the chromosome at
  `integration_locus`. Applying the `GeneAddition` edits the chromosome sequence. The
  delivery/CRISPR plasmids themselves are cured -> NOT stored as cell contigs.

Corollary: only RETAINED genetic features become `GeneAddition`s. Transient CRISPR/sgRNA
vectors (Cachera pHO8/22/25/29) are neither content nor perturbations -- they leave no
trace in the final cell sequence.

### Applied to the two motivating datasets

- **Cachera Btx-cassette = chromosomal_integration.** pBTX1/pBTX2 are CONSTRUCTION
  sources; the cassette integrates at chromosome **XII-5**. From `pBTX002.gb` the insert
  is the segment between `XII-5\UP` (1548-2002) and `XII-5\Down` (10106-10782) and holds
  the four genes with promoters/terminators: **ARO7-G141S** (771 bp, TEF1p), **ARO4-K229L**
  (1113 bp, PGK1p), **CYP76AD1** (1500 bp, as a P450 fusion, TPI1p), **DOD** (810 bp,
  CCW12p). Not a present-plasmid contig.
- **Ozaydin YB/I/BTS1 = episomal_2micron (present).** YEplac195 2-micron URA3 plasmid,
  retained under SC-URA selection during the color screen -> a genuine present contig.
  But no sequence exists (Euroscarf P30796 physical-only) -> reconstruct-from-parts
  before it can be stored.

### What was built (this branch)

`torchcell/sequence/plasmid.py`: pydantic `PlasmidSequence` (id/topology/length/sequence
/features + `PlasmidProvenance` sha256) with `parse_plasmid_genbank` (BioPython SeqIO)
and `feature_sequence(label, flank)` -- coding-strand extraction (reverse-complements
minus-strand features; `flank` adds +/- promoter/terminator context; wraps a circular
plasmid). Verified: all 17 mirrored Cachera GenBank maps parse; the four pBTX2 cassette
genes extract with correct coding-strand sequences (each begins with ATG). 7 unit tests
(forward/reverse/flank/circular-wrap/missing/duplicate/round-trip); ruff + mypy clean.

Mirror: `$DATA_ROOT/torchcell-library/cacheraCRISPAHighthroughputMethod2023/si/plasmids/`
(pBTX1=`pBTX001(ante113).gb`, pBTX2=`pBTX002.gb`, + CRI-SPA helpers), each hash-pinned in
`gkad656_supplemental_files.provenance.json`.

### Blockers / next (NOT in this branch)

- **GeneAddition wiring** (`plasmid_contig_id`/`locus_tag` -> this store, or an
  integration-insert reference) waits on `ws8` (GeneAddition) landing.
- **Apply-to-sequence**: the function that materialises the in-cell sequence (base
  genome + KO edit + integration insert + present plasmid contigs) from a `Genotype` --
  the consumer the embedding datasets call.
- **Ozaydin 2-micron reconstruction** (YEplac195 backbone + crtYB/crtI/BTS1 + TDH3p/CYC1t)
  so its present plasmid can be stored.
- Persisted store format (LMDB/JSON) + ingest CLI over the mirror; currently parse-on-demand.
