---
id: cyi1htfjmy59b2dzm9jruud
title: Caudal2024
desc: ''
updated: 1783474258103
created: 1783474258103
---

## 2026.07.07 - WS10 pan-transcriptome design (Caudal 2024 + Peter 2018)

Roadmap WS10. Caudal et al. 2024, *Nature Genetics* 56(6):1278-1287,
DOI `10.1038/s41588-024-01769-9`, key `caudalPantranscriptomeRevealsLarge2024`.
Design informed by three scouts + the reconstructed-PDF MinerU OCR (`paper.md`) +
direct inspection of the downloaded matrices.

### The core reframing: a population survey modeled as perturbations off S288C

Every prior torchcell dataset is `perturbation(s) -> phenotype` on the S288c reference
background (a KO/edit set is the genotype). The pan-transcriptome is different in kind:
**969 natural isolates**, each with its OWN complete genome (SNPs, gene CNVs, introgressions,
HGT, aneuploidy/ploidy 1N-5N vs S288c). So:

- **Genotype = a perturbation SET off the S288C reference** (decision 2026.07.07, w/ user).
  A natural isolate is NOT modeled as a bespoke "whole genome" object and NOT as a
  reference of its own -- it reuses the SAME perturbation ontology as an engineered strain,
  because "perturbation = an edit to the total genomic content off a WT" is exactly what a
  natural isolate is, at scale. Three perturbation kinds, all off S288C R64:
  - **core gene differing by SNPs/indels -> an allele-variant perturbation** carrying a
    POINTER to that isolate's gene sequence in the off-graph gene-keyed store (never the
    amino-acid-substitution `AllelePerturbation`; a new sequence-level variant type);
  - **accessory ORF present (not in S288C) -> `GeneAdditionPerturbation`** (the ws8 type),
    pointing at the pangenome ORF sequence;
  - **core ORF absent in the isolate -> `DeletionPerturbation`**.
  S288C stays the shared annotation/coordinate frame (`genome_reference`); the isolate's
  divergence is the genotype. Sequences are NEVER inlined -- perturbations carry off-graph
  pointers (gene id + source + sha256), dereferenced at load (north-star "genomes are
  pointers"). This is the general, scalable pattern the whole schema is built on.

  FEASIBILITY (measured from Peter data, 2026.07.07): per isolate ~4,500 allele-variants
  (70% of isolates differ from the modal/reference allele at a typical gene) + ~519
  GeneAdditions (median accessory present; max 736) + ~0-84 Deletions (core is highly
  conserved: 5,491 core / 2,305 accessory of 7,796 pangenome ORFs) ~= ~5,000
  perturbations/isolate, ~4.8M objects across 969 expressed isolates -- tractable ONLY as
  pointers (969 records, each a ~5k-pointer genotype; sequences off-graph).
- **Phenotype = absolute expression** per (isolate, gene): TPM + raw count. There is NO
  `log2(sample/reference)` ratio -- the microarray family's convention does not apply.
  Sourced from `paper.md` L55: abundance = "mean log2 of the normalized read counts
  (transcripts per million (TPM))". We store the raw TPM + count; log2/abundance/dispersion
  are downstream derived metrics, not the stored value.

### Sourced facts (from OCR `paper.md` + data)

- 1,032 isolates RNA-seq -> **969 high-quality** transcriptomes (QC threshold: >= 1 million
  mapped reads) (`paper.md` L25). 26 clades. 29 culture replicates (avg r = 0.94, L25).
- **6,445 transcripts = 4,977 core + 1,468 accessory** ORFs (`paper.md` L27). Accessory =
  variably present across isolates; for an isolate lacking a gene, that gene is absent (NOT
  zero) -- must be encoded as missing, not 0 TPM (L55: "isolates that did not carry the
  given gene were excluded").
- Genomes "previously completely sequenced" = **Peter et al. 2018** (1,011 Yeast Genome
  Project, `peterGenomeEvolution10112018`, DOI `10.1038/s41586-018-0030-5`) -- the genome
  source, a DIFFERENT paper. Genomes are NOT in Caudal.

### Raw data (mirrored + sha256-pinned, 2026.07.07)

`$DATA_ROOT/torchcell-library/caudalPantranscriptomeRevealsLarge2024/`:

- `paper.pdf` (main text, 9 pages, reconstructed from Swanki per-page PDFs; sha256
  `4394f51a...`; canonical nature.com PDF is a pending manual deposit) + `paper.md` (MinerU
  OCR, 350 DPI).
- `data/final_data_annotated_merged_04052022.tab.zip` (Datafile 1; sha256 `8b55ccd7...`;
  111,907,574 B; unzips to 901 MB). **COMMA-delimited despite `.tab`.** Long format, one row
  per (Strain, gene): key columns `systematic_name, ORF, Strain, count, tpm,
  Pangenome(Core/Accessory), Group, Precence_in_S288c`. The core expression matrix.
- `data/replicate_data_tpm_22042023.tab` (Datafile 2; sha256 `6860fa5f...`; 78 MB). Replicate
  TPM (29 reps) + rich per-strain metadata (Standardized.name, YJS.name, clade/Group, mapped
  reads, OD/midlog filtration, ecological/geographical origin, ploidy).
- NOT retrieved: `ASE_data_counts.csv` (allele-specific, Datafile 3), GWAS `.tab` (eQTL,
  Datafile 4) -- downstream analyses, not raw expression.
- Full data availability + accessions (ENA `PRJEB52153` raw reads; GitHub/Zenodo code) in the
  mirror `manifest.json`.

### Schema: new `RNASeqExpressionPhenotype` (additive; do NOT overload microarray)

Sibling of `MicroarrayExpressionPhenotype` in the expression family (`schema.py`). Carries
absolute NGS expression, not a ratio. Proposed fields (all per-gene dicts keyed by
systematic ORF, coerced to `SortedDict` like the microarray class):

- `expression_tpm: dict[str,float]` (primary), `expression_count: dict[str,int]` (raw),
- `n_mapped_reads: int` (per-isolate QC; from Datafile 2 metadata where available),
- `measurement_type: str = "rnaseq_tpm"`, plus core/accessory presence handled by KEY
  ABSENCE (a gene absent from the isolate's genome is simply not a key -- honest to L55).

Reference: this is absolute expression, so `reference_centered=False` semantics (like
Mülleder / the metabolite family). Whether to also emit a population-level reference
(e.g. S288c or median-isolate TPM) is an OPEN decision -- default: no reference profile;
store absolute values only, and let downstream centering be a transform.

### Off-graph sequence store + the genotype-from-perturbations model

The genotype is a perturbation set off S288C (above); the isolate genome is NOT stored as a
separate object -- it is RECONSTRUCTIBLE as `apply(S288C reference, isolate perturbations)`
(the deferred apply-to-sequence function). Perturbations carry off-graph POINTERS; the
store holds the sequences. Confirmed Peter 2018 data layout (all mirrored + hash-pinned):

- **gene-keyed per-isolate variant store** = `allReferenceGenesWithSNPsAndIndelsInferred.tar.gz`:
  gene-major, one `Y<sys>.fasta` per S288C reference gene (6,015 files), each holding all
  1,011 isolates' copy of that gene (header `SACE_<ISO>_<sys>_<symbol>  chrN:start-end strand`
  -- S288C-coord-anchored). An allele-variant perturbation points here by (systematic gene id,
  isolate id); the reference allele = our `SCerevisiaeGenome` gene, so "differs" = isolate seq
  != reference seq (NO S288C record inside the file). ~70% of isolates differ at a typical gene.
- **pangenome accessory-ORF store** = `allORFs_pangenome.fasta.gz` (7,796 ORFs; headers like
  `1-EC1118_1F14_0012g`) + `genesMatrix_PresenceAbsence.tab.gz` (1,011 isolates x 7,796 ORFs,
  0/1) + `genesMatrix_CopyNumber.tab.gz`. A GeneAddition points at a pangenome ORF sequence;
  presence=1 for a non-core ORF -> addition, presence=0 for a core ORF -> deletion.
- **draft assemblies** = `1011Assemblies.tar.gz` (3.8 GB; per-isolate de novo FASTA, contig-
  level, N50 ~136 kb, ~3,259 contigs, NO per-strain GFF) -- archival mirror only, NOT a load
  path (unannotated draft contigs; the gene-keyed store is what feeds embeddings).

NEW schema needed (additive):

- a **sequence-level allele-variant perturbation** (working name `AlleleSequencePerturbation`)
  carrying `systematic_gene_name` + off-graph pointer fields (`sequence_uri`/`gene_fasta`,
  `strain_id`, `sha256`; sequence never inlined) -- distinct from the aa-substitution
  `AllelePerturbation`. Mirrors the `GeneAddition.plasmid_contig_id`/`locus_tag` pointer
  pattern (ws8). Register in the perturbation unions.
- reuse `GeneAdditionPerturbation` (ws8) for accessory-present (source_organism = S. cerevisiae
  or the introgression/HGT donor when known; localization = the isolate genome), and
  `DeletionPerturbation` for core-absent.

Strain-code join (confirmed): Caudal Datafile 1 `Strain` (3-letter, e.g. AAB) == the 1002-
project codes used by Peter's presence/absence rows and gene-file headers; Caudal Datafile 2
carries `Standardized.name`/`YJS.name` for any prefix reconciliation (e.g. `SACE_YAM` vs bare
`BFC`). Build the code<->code map once from Datafile 2 + the Peter matrices.

### Build plan / status

1. [done] OCR Caudal + Peter; mirror + hash-pin Caudal Datafiles 1-2 and Peter gene-keyed +
   matrices + assemblies; source normalization/counts; measure the perturbation-load feasibility.
2. [done, this branch] `RNASeqExpressionPhenotype` schema + invariant tests (additive).
3. [next] `AlleleSequencePerturbation` schema (+ off-graph pointer fields) + register; the
   off-graph gene-keyed + pangenome sequence store (URI + sha256 + lazy dereference), shared
   with the plasmid store's pending persisted layer.
4. [after 3] `CaudalPanTranscriptome2024Dataset` loader: build the strain-code map; per isolate
   diff its 6,015 gene sequences vs the `SCerevisiaeGenome` reference -> allele-variant
   perturbations (pointers); read presence/absence -> GeneAddition/Deletion; assemble the
   ~5k-pointer genotype (off S288C); phenotype = `RNASeqExpressionPhenotype` from Datafile 1
   (absolute TPM+count; accessory absent = key-absent); genome_reference = S288C R64;
   phenotype_reference = None; strain metadata from Datafile 2; 969 records; L0-L4 runner spec.

Coordination: this branch touches the expression-family region + new perturbation type +
new sequence-store files; the Zelezniak-metabolite branch (landed) touched only the
`MetabolitePhenotype` family -- disjoint. See
`[[torchcell.sequence.plasmid-and-genomic-content-design]]` and
`[[plan.schematization-ingestion-roadmap.2026.06.23]]` (WS10).
