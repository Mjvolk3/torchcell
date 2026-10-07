---
id: oeeyfjirt1ks0kt7dn3q9ds
title: Bacterial
desc: ''
updated: 1791374701422
created: 1791374701422
---


## 2026.10.07 - Shared layer for NCBI-annotated bacterial genomes

Module `torchcell/sequence/genome/bacterial.py`, added in step 3 of
[[plan.bacteria-ontology-genome]] so that [[torchcell.sequence.genome.ecoli.k12]] and
[[torchcell.sequence.genome.pputida.kt2440]] share one implementation instead of two
copies. Base: [[torchcell.sequence.genome.base]]. Tests:
[[tests.torchcell.sequence.genome.test_bacterial]] (parsers) and the two host test
modules (genomes).

### Pieces

- **Records (pydantic).** `GenBankLocus`, `GenBankReplicon`, `GenBankAnnotation`;
  `RefSeqGffAnnotation`; `GafSynonymGo`; `GoSourceSpec` (route, assembly set, member,
  identifier pattern); `GoAnnotationSource` (what the route read and reached, measured
  at construction); `BacterialAssembly` (one assembly as a class reads it, with derived
  member names).
- **Readers.** `read_genbank` (loci plus the CDS of each coding locus, joins included);
  `read_protein_fasta` (protein FASTA re-keyed from accession to locus tag);
  `read_refseq_gff` (RefSeq tag to `old_locus_tag`, and `Ontology_term` rows);
  `read_gaf_synonym_go`; `refseq_inline_go`. Each refuses an inconsistent file by name:
  an orphan product feature, a repeated tag, a non-unique product, a missing protein, a
  GAF row without 17 columns, a malformed GO id, RefSeq rows that disagree on
  `old_locus_tag`.
- **Product rule.** A locus's product is the one product feature (CDS, tRNA, rRNA,
  ncRNA, tmRNA, misc_RNA, precursor_RNA) whose span equals the gene's. Other CDS on the
  locus are isoforms (MG1655 has 9 such loci, e.g. `mrcB`'s PBP-1Bgamma) and contribute
  only `isoform_protein_ids`. On the three deposited assemblies every non-pseudo gene
  has exactly one spanning product except MG1655 `b2621`, which has none.
- **`BacterialGenome`.** `release_files` names the GCA FNA, GFF and FAA with
  `cds_fasta=None`; `_read_sequences` parses the GenBank file, checks the replicon and the
  locus-tag pattern, and reads the gzipped FASTAs; construction then checks `data.db`
  against the GenBank loci, reads the RefSeq GFF, and loads GO by the assembly's route.
  `compute_gene_set` is the GenBank non-pseudo tags. `__getitem__` returns None for a tag
  that is neither a current gene nor a pseudogene (an unknown tag, or one dropped by
  `drop_empty_go`) and refuses a joined pseudogene, which one interval cannot represent.
  GO is held in memory (`go_annotations`), never written into `data.db`, so
  `remove_deprecated_go_terms` filters it in memory. `locus_table` is one row per
  GenBank locus.

### Name resolution

Order: exact locus tag (CURRENT for a gene, NON_GENE_FEATURE for a pseudogene),
`old_locus_tag` (none in the three GCA files), RefSeq locus tag through its
`old_locus_tag`, gene symbol, `gene_synonym`, then RETIRED with the name kept as given.
Status semantics in layers 2 to 5 are the base resolver's (unique gene RENAMED, several
AMBIGUOUS, unique pseudogene NON_GENE_FEATURE). The bacterial resolver overrides the base
one for two reasons, both measured:

- The base resolver upper-cases its input and returns that (`YAL001C`); bacterial tags
  are mixed case (`b0002`, `PP_t01`), so a CURRENT result must return the annotation's
  own case. Changing the base would change S288C results for mixed-case SGD loci
  (`tA(UGC)A`), so the base is untouched.
- Exact case first, then case-insensitive only when no exact match exists. Case folding
  alone would make `Pro2` (proC) and `pro2` (proB) in MG1655, and `flrD`/`flrd`,
  `tabB`/`TabB` in BW25113, ambiguous; with exact-first they resolve, and `PRO2` is
  AMBIGUOUS with a note saying the match was case-insensitive.

The symbol layer runs before the synonym layer because 49 MG1655 synonyms (88 in
BW25113) equal another locus's symbol.

### Hooks added to the base for this layer

Three gaps, each behavior-preserving for S288C (its tests pass unchanged):
`AnnotatedGene.FEATURE_ID_PREFIX` (NCBI writes `ID=gene-b0002`; default `""`), honored by
the gene lookup, `compute_gene_set`, `feature_index` and `drop_empty_go`;
`GenomeReleaseFiles.cds_fasta` may be `None` (NCBI GCA sets ship no CDS FASTA); and the
`_read_sequences` hook (the GCA FASTAs are gzipped and the proteins keyed by accession).
See the dated section in [[torchcell.sequence.genome.base]].
