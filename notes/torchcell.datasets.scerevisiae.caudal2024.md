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

### The core reframing: a population survey, not a perturbation screen

Every prior torchcell dataset is `perturbation(s) -> phenotype` on the S288c reference
background (a KO/edit set is the genotype). The pan-transcriptome is different in kind:
**969 natural isolates**, each with its OWN complete genome (SNPs, gene CNVs, introgressions,
HGT, aneuploidy/ploidy 1N-5N vs S288c). So:

- **Genotype = the whole natural-isolate genome**, NOT a perturbation list. This is the
  purest expression of the CLAUDE.md "total genomic content in the cell" principle -- the
  genotype IS the strain's sequence. It is represented as a POINTER to an off-graph genome
  assembly (see off-graph design below), never as an enumerated variant set.
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

### Off-graph per-strain genome store (the NEW subsystem)

`genomes are pointers, never payloads` (roadmap decision 6). NONE of this exists yet (scout
confirmed: zero `GenomeAssembly`/`SequenceRecord`/`fasta_uri` hits in code). Build surface:

- NEW pydantic pointer nodes: `GenomeAssembly` / `SequenceRecord` / `GenomeFeature`
  (`assembly_id, strain, species, fasta_uri, gff_uri, sha256, version`). These parallel the
  plasmid store's `SequenceProvenance` (`torchcell/sequence/plasmid.py`) but add the
  URI-dereference indirection the plasmid store still lacks (it holds sequence inline).
- NEW `PerStrainGenome(Genome)` implementing the existing `Genome`/`Gene` ABC
  (`torchcell/sequence/data.py`) -- once it implements `compute_gene_set` / `get_seq` /
  `__getitem__` / `window*`, the WHOLE embedding pipeline (nucleotide_transformer, esm2,
  protT5, codon_frequency, one_hot) reuses it unchanged. Mirrors `SCerevisiaeGenome`'s
  gffutils + SeqIO parsing.
- NEW off-graph persisted store + lazy-dereference loader (read FASTA/GFF by URI, verify
  sha256, materialize in-memory). Build this ONCE, shared with the plasmid store (its design
  note already lists a persisted/URI layer as pending).
- Genome source = Peter 2018 assemblies (1002genomes host + ENA). **BLOCKED on**: (a) the
  Peter 2018 PDF landing in the Zotero group so we can OCR the Methods + confirm the exact
  assembly accession/URL, (b) mapping Caudal 3-letter `Strain` codes <-> Peter 2018
  `Standardized.name`/`YJS.name` (Datafile 2 carries both -- the join key).

### Build plan / status

1. [done] OCR paper; mirror + hash-pin Datafiles 1-2; source normalization + counts.
2. [this branch] `RNASeqExpressionPhenotype` schema + invariant tests (additive, low-risk).
3. [pending Peter 2018] off-graph genome pointer nodes + `PerStrainGenome` + persisted store.
4. [after 2+3] `CaudalPanTranscriptome2024Dataset` loader: stream Datafile 1 (chunked; 6.2M
   rows), genotype = `GenomeAssembly` pointer (fasta_uri=None until Peter lands, mirroring
   the `GeneAddition.plasmid_contig_id=None` precedent), phenotype = `RNASeqExpressionPhenotype`;
   strain metadata from Datafile 2; L0-L4 via a new `run_expression`-style runner spec.

Coordination: this branch touches ONLY the expression-family region of `schema.py` + new
files; the concurrent Zelezniak-metabolite branch touches only the `MetabolitePhenotype`
family -- no shared class edited by both (scout-confirmed). See
`[[torchcell.sequence.plasmid-and-genomic-content-design]]` and
`[[plan.schematization-ingestion-roadmap.2026.06.23]]` (WS10).
