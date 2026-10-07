---
id: avsa920upj04qact71qsxj3
title: Test_bacterial
desc: ''
updated: 1791374725436
created: 1791374725436
---


## 2026.10.07 - Parser tests on synthetic NCBI files

Covers [[torchcell.sequence.genome.bacterial]]'s readers. Fixtures come from
`tests/torchcell/sequence/genome/_bacterial_fixtures.py` (Biopython writes the GenBank
file; GFF3, FASTA, GAF and OBO are written as text, gzipped like NCBI's). Pins: every
`GenBankLocus` field (`;`-split synonyms, isoform protein ids, pseudo flags, joined
segments), the CDS extraction (minus strand reverse-complemented), the protein re-keying,
the GAF token rule (`/` split, `b0005.1` never read as b0005, NOT rows excluded), the
RefSeq `old_locus_tag` crosswalk and inline GO, and each named refusal. Runs everywhere;
reads no tier.
