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

## 2026.07.07 - Built `CaudalPanTranscriptome2024Dataset` (943 isolates, L0-L4 PASS)

`torchcell/datasets/scerevisiae/caudal2024.py` -- built + verified. Each natural isolate
is modeled as a PERTURBATION SET off the S288C reference, with the whole-transcriptome
expression as its phenotype.

### Records + per-isolate genotype composition

- **943 records** = the intersection of Caudal `Strain` codes (969) with Peter's genome
  panel (1011 isolates via `genesMatrix_PresenceAbsence.tab.gz` index). The 26 Caudal-only
  strains have **no Peter genome and are EXCLUDED**: 25 `XTRA_*` (DCN, DCO, DCP, DCT, DCU,
  DCV, DCW, DCZ, DDA, DDB, DGH, DGR, DGT, DGU, DGW, DGX, DGY, DHB, DHD, DHE, DHJ, DHK, DHO,
  DHQ, DXL) + `FY4-6`.
- **Per-isolate genotype means** (over the 943): `sequence_variant` = **5047.3**,
  `copy_number_variant` accessory-present = **519.2**, `copy_number_variant` core-loss =
  **1.59**. Total ~5568 perturbations/isolate; 4,759,608 sequence variants diffed overall.
- Example (`dataset[0]`): 4557 sequence variants + 457 CNVs; phenotype 6000 genes; genome
  reference `{Saccharomyces cerevisiae, S288C}`.

### Reference-slice method (sequence variants, VALIDATED)

For each of Peter's **6015** gene-keyed FASTAs (`allReferenceGenesWithSNPsAndIndels
Inferred.tar.gz`, sha256 `b5400b89...`), the coordinate in the FIRST record header
(`chromosomeN:start-end +/-`) slices the SGD R64 chromosome (`[start-1:end]`, reverse-
complemented on the minus strand). That slice IS the S288C reference allele in Peter's
exact representation (6-165 of the 1011 isolates match it verbatim per gene). An isolate
has a `SequenceVariantPerturbation` at that gene iff its uppercased sequence != the slice.
The isolate id is recovered by splitting the header token on `_<SYS>_` (matched isolates
are the 3-letter Caudal codes). `sequence_uri = "<SYS>.fasta#<header_token>"`,
`sequence_sha256 = b5400b89...`. All 6015 gene names are valid S288C systematic names, so
`SequenceVariantPerturbation`'s (non-relaxed) name validator never rejects.

### CNV method (pangenome presence/absence)

Peter's presence matrix (1011 x 7796 ORFs) classifies each pangenome ORF **core** (present
in >= 99% of isolates -> 5491) vs **accessory** (2305). Matrix columns are R-`make.names`-
mangled (`X1834.YAL063C`); demangling (drop leading `X`, first `.`->`-`) recovers the raw
ORF id (`1834-YAL063C`), whose `<number>-<name>` suffix is an S288C systematic name for
reference ORFs (5276 of 7796 map to S288C; 1:1, no collisions) and an assembly id
otherwise.

- **Accessory PRESENT** -> `CopyNumberVariantPerturbation` (`copy_number` from the copy-
  number matrix, default 1.0 if 0/NaN; `reference_copy_number = 0.0`;
  `systematic_gene_name = pangenome_orf_id`).
- **Core ABSENT** -> `CopyNumberVariantPerturbation` (`copy_number = 0.0`,
  `reference_copy_number = 1.0`, `systematic_gene_name =` the S288C name) **only when the
  ORF maps to an S288C name** (4807 of 5491 core ORFs do); core-loss of an unmappable ORF
  is skipped (hence the low 1.59 mean -- most isolates lose 0-2 core ORFs mappable to
  S288C).

### Phenotype + population-mean reference decision

`RNASeqExpressionPhenotype`: per-isolate `expression_tpm` + `expression_count` (raw reads,
`int(round(sum))`) for the genes that isolate carries in Caudal
`final_data_annotated_merged_04052022.tab` (comma-delimited, latin-1; sha256 `8b55ccd7...`).
A gene absent from an isolate is KEY-ABSENT (never 0). The merged file already has one row
per (Strain, gene) (defensive sum-aggregation is a no-op). `measurement_type = "rnaseq_tpm"`,
`n_mapped_reads = None`. **`phenotype_reference` = the POPULATION MEAN over the 943 built
isolates** (mean TPM / rounded mean count per gene; one shared object reused across all
records) -- an absolute WT-equivalent baseline, so verification uses `reference_centered =
False` (checks finiteness, not a centered 0). `genome_reference = {S. cerevisiae, S288C}`;
environment `SC` liquid, 30 C (Caudal Methods: mid-log OD ~0.3). Publication PMID 38862621
/ DOI 10.1038/s41588-024-01769-9.

### L0-L4 verification (`torchcell/verification/rnaseq.py` + `run_rnaseq`) -- PASS

```text
caudal_pantranscriptome2024: PASS
  [ok] L0 structural: 943 records validated
  [ok] L1 count: observed 943, expected 943
  [ok] L1 strain_uniqueness: 943 unique isolates, one record each
  [ok] L2 tpm_value_fidelity: 5685741 values checked
  [ok] L2 count_value_fidelity: 5685741 counts are non-negative integers
  [ok] L3 measurement_type_consistent: single measurement_type: 'rnaseq_tpm'
  [ok] L3 reference_finite: reference TPM finite for all 6086122 values
  [ok] L4 gene_containment_sgd: 0.943 of 6454 measured genes are S288C reference genes (>= 0.9)
```

The 5.7% of measured genes outside the SGD ORF+RNA set are accessory/novel pangenome ORFs
(`EC1118_*`, `maker.*`, `snap_masked.*`) legitimately absent from S288C. The heavy step is
the 6015-gene x 1011-isolate sequence diff (~a few min; cached to
`preprocess/sequence_variants.parquet` for resumable re-assembly). Verification re-validates
all ~5.7M perturbations through the schema union, so `run_rnaseq` itself takes ~20 min.

## 2026.07.15 - Fix: gene-absence edits were silently dropped (#71)

`_content_perturbations` gated both loops on **population frequency** (`core_mask`, present
in >=99% of isolates) instead of **S288C reference membership** (`s288c_mask`). A reference
ORF that is *variable* (not core) and *absent* from an isolate matched neither loop, so **no
perturbation record was emitted** and the isolate reconstructed as if it still carried the
gene -- the exact failure the "never dropped" invariant forbids.

Measured impact (audit vs the released presence/absence matrix): median **0** absence
records emitted per isolate vs **126** that should be, **134,428** missing gene-absence
edits across the 1,011-isolate panel.

**Fix (both guards now on `s288c_mask`):**

- Reference ORF ABSENT -> `NaturalGeneAbsencePerturbation`.
- Non-reference (accessory) ORF PRESENT -> `NaturalGenePresencePerturbation`.
- A reference ORF present and an accessory ORF absent are both no-ops vs S288C.
- `_orf_to_s288c` now strips the `_NumOfGenes_N` paralog-cluster suffix, recovering **793**
  reference ORFs it previously returned `None` for (5,276 -> 6,069 reference columns; every
  reference name maps from exactly one column, so no de-dup needed).

**Rebuilt** 2026-07-15 (cached `sequence_variants.parquet` reused): 943 records; record 0
perturbation-type counts `natural_gene_absence` **2 -> 163**, `natural_gene_presence`
**511 -> 48** (the ~451 removed were *present variable-reference* ORFs the old code
mislabeled as accessory presences under their pangenome id, not `YAL005C`), `sequence_variant`
4,557 (unchanged). Phenotype (expression TPM) unchanged, so the L1-L4 expression checks are
unaffected.

**DEFERRED -- reference-ORF dosage (CNV).** The old presence path was also (buggily) the
only place reference-ORF copy-number reached the record; the clean fix drops it. ~102
present reference ORFs/isolate carry |CN-1|>0.5. Representing this properly needs
`CopyNumberVariantPerturbation` + a dosage threshold, and **Peter 2018 gives no citable CNV
cutoff** (raw coverage-ratio only), so per the never-guess-a-threshold rule this is a
follow-up, not part of this fix. Sequence variants (~4,557/isolate) dominate the genotype,
so the headline genotype-information counting is ~unchanged.

**Downstream:** the genotype-side 018 analyses (`bit_accounting`, `verify_signal_composition`)
and the `Signal (gzip)` genotype block change (fewer, differently-typed perturbations);
`differential_expression_comparison` / expression figures are phenotype-based and unchanged.

**Note on rebuild hygiene:** a partial rebuild that keeps `preprocess/` must also clear the
stale `experiment_reference_index.json` (pre-WS15 dense format, 2026-07-10 redesign) or
`ExperimentReferenceIndex.from_stored` raises; a full `processed` + regenerable-`preprocess`
clear (keeping only `sequence_variants.parquet`) rebuilds cleanly.
