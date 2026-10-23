---
id: 9xmvzlzhx51p9crgg7q90hu
title: Compute_isolate_composite
desc: ''
updated: 1792634724979
created: 1792634724979
---

## 2026.10.21 - Composite allele embeddings for all 1,011 isolates: CPU stages run, GPU stage held

Script: `experiments/043-expression-proteome-morphology/scripts/compute_isolate_composite.py`.
Launchers: `gh_compute_isolate_composite_cpu.slurm` (stage as first argument) and
`gh_compute_isolate_composite_embed.slurm` (GPU, three cheap encoders then ProtT5, then verify).
Store: `$DATA_ROOT/data/scerevisiae/isolate_embeddings/peter2018_1011/` (`manifest.json` is a
pydantic `IsolateEmbeddingManifest` carrying the full `IsolateCompositeConfig`, the encoders,
the window geometry, the conventions, tool versions, dedup counts and one `StageRecord` per
stage). Derived GFF3 plus a pydantic `IsolateAlignmentRecord` per isolate under
`$DATA_ROOT/torchcell-genomes/peter2018_1011_assemblies/derived/<isolate>/`.

The composite is the trigenic case study's, `node_embeddings: [fudt_upstream, calm,
prot_T5_all, fudt_downstream]` (`experiments/030-solid-growth-multi/conf/cgt_030_s3_r_tok_embfit_003.yaml`).
Every table is keyed by the sha256 of the exact string its encoder reads, so an identical allele
is one row, and the S288C rows the composite trained on are copied in (the `seed` stage) so an
isolate allele identical to S288C reuses the trained-on vector.

### Scope

All 1,011 Peter isolates (the owner's "all alleles"; the subset would have saved about 3% of a
13 minute extract and 44 CPU-minutes of BLAST). The store's 979 isolates are a subset by
construction: `natural_isolate.assert_all_isolates_seen` makes every isolate loader fail if a
built isolate is missing from the tarball headers. The five non-BY segregant parents are in it:
`bloom2019.PARENT_PETER_ID` (imported by `boocock2025.py` and `nguessan2025.py`) gives
RM11-1a = AAA, YJM145 = AAS, YPS163 = ACD, CBS2888 = ABL, YJM981 = ADI, all five in the tarball
headers and in the assembly tarball (`AAA_6.re.fa`, `AAS_3.re.fa`, `ACD_7.re.fa`, `ABL_5.re.fa`,
`ADI_4.re.fa`).

### Stages

| stage | what | resources | job | wall | peak RSS |
|---|---|---|---|---|---|
| `extract` | tarball pass: allele span, spliced CDS, protein, CaLM input, hashes, R64 position, presence and copy number; accessory ORFs; per-isolate query FASTA | 2 CPU, 48 GB | 6052 | 12:41 | 11.7 GB |
| `unpack` | stream `1011Assemblies.tar.gz`, write each member with its sha256 | 2 CPU, 8 GB | 6053 | 1:07 | 1.3 GB |
| `align` | megablast each isolate's queries against its own assembly; locus, flanks, windows, derived GFF3 | array 0-31%16, 4 CPU, 4 GB | 6071 (tasks 6071-6105) | 1:39 max task | 1.5 GB (timing run 6069) |
| `seed` | copy the four S288C tables in, keyed by the string each row actually embedded | 8 CPU, 24 GB | 6088 | 2:27 | 2.3 GB |
| `merge` | join, distinct window tables, S288C reference context, dedup counts | 2 CPU, 64 GB | 6089 | 1:37 | 16.5 GB |
| `verify` | every row resolves or carries a typed gap; per-isolate counts | 2 CPU, 48 GB | 6090 | 0:33 | 11.2 GB |
| `check` | batched encoder path against the S288C tables, CPU, nothing written | 16 CPU, 24 GB | 6050, 6054, 6070 | 0:36 to 2:08 | |
| `embed` | GPU, distinct inputs not yet in each table | 1 card, 6 CPU, 48 GB, 18 h | 6106, HELD | not run | |

Align: 1,010 isolates aligned in the array plus AAC in the timing run 6069; per isolate
median 2.5 s, max 4.1 s, 2,612 CPU-seconds in all (`work/align_logs/`).

### Counts (measured)

From the manifest stage records of jobs 6052, 6089 and the verify report of job 6090:

- Rows: 6,165,757 = 6,081,165 reference-gene rows (1,011 isolates x 6,015 genes, every isolate
  complete) + 84,592 accessory-ORF rows (1,649 distinct ORFs; 34 to 352 per isolate, median 78).
- Reference-gene state from the presence matrix: present 5,618,191, absent 117,212, gene has no
  matrix column 345,762.
- Gene models: 6,015 from the gene-file headers; 239 multi-exon maps and 25 single-interval
  maps (an SGD CDS that is a sub-span of the Peter span) from the R64-4-1 GFF; 28 genes whose
  SGD gene span differs from the Peter header span; 4 tarball genes not in the SGD gene set
  (YER109C, YFL056C, YFL057C, YOR031W); 2 models whose exon map does not reproduce the SGD CDS
  from the R64 span (YGL033W, YMR147W), giving 2,022 typed-gap rows (`splice_unverified`).
- Splicing of distinct alleles: 1,154,177 contiguous, 28,976 cut at the reference offsets,
  15,031 carried through a global alignment (indel alleles of intron genes), 52 rows whose exon
  boundary fell in an alignment gap (`exon_boundary_in_alignment_gap`).
- Loci (all rows): located 5,798,925; partial alignment 237,818; below 95% identity 97,523; no
  hit 28,081; CDS off the contig end 3,410. Present reference genes only: 5,445,076 of
  5,618,191 located (96.9%), partial 153,483, below identity 17,769, off contig 1,732, no hit 131.
  Lowest located: BFG 4,321, BTE 4,328, CBF 4,390 (median isolate 5,804).
- Windows: upstream 5,709,125 rows (typed gaps 366,832 locus + 89,800 short flank); downstream
  5,756,614 (366,832 locus + 42,311 short flank).
- Reference position (R64 chromosome, start, end, strand, header order) on all 6,081,165
  reference-gene rows; accessory ORFs carry none.

### Dedup counts (measured, merge job 6089)

| input | distinct over isolates | with the S288C context | seeded from the S288C table | still to compute |
|---|---|---|---|---|
| ProtT5 protein | 757,124 | 757,720 | 6,536 | about 751,000 |
| CaLM input | 1,162,730 | 1,163,337 | 6,558 | about 1,157,000 |
| FUDT upstream 1,003 bp | 1,137,075 | 1,137,931 | 6,571 | about 1,131,000 |
| FUDT downstream 300 bp | 412,926 | 413,606 | 6,557 | about 407,000 |

Distinct CDS 1,183,044 and distinct gene spans 1,200,150. The earlier
`count_distinct_alleles.py` totals (1,198,539 CDS, 775,856 proteins) translated the unspliced
span of intron genes and left out accessory ORFs, so they differ. The primer's hypothesis that
the window dedup ratio is close to the CDS ratio holds for the upstream window (1.14 M vs 1.18 M
CDS) and not for the downstream window (0.41 M).

### GPU estimate (hypothesis, rates not measured)

Cards are held by the 031 packs, so no rate was measured. Assumed rates on one RTX 6000 Ada:
FUDT upstream fp32 150 to 300 per s (1.0 to 2.1 h), CaLM fp32 300 to 600 per s (0.5 to 1.1 h),
FUDT downstream fp32 600 to 1,000 per s (0.1 to 0.2 h), ProtT5-XL fp16 30 to 60 per s (3.5 to
7 h): 5 to 10.5 h in all. Job 6106 asks for 18 h; it appends every 200 batches and a resubmit
skips done rows. Storage at fp16: about 1.5 GB ProtT5, 1.8 GB CaLM, 1.7 GB FUDT upstream, 0.6 GB
FUDT downstream.

### Encoder reproduction (measured)

The batched path groups inputs by exact token length, so no batch pads and a mean over tokens is
the per-sequence mean. Against the stored S288C rows, 16 genes each, CPU: FUDT upstream min
cosine 0.99999988, max abs diff 3.8e-6 (job 6050); CaLM min cosine 0.99999994 (6054); FUDT
downstream max abs diff 0.0 (6054); ProtT5 fp32 vs the fp16 table min cosine 0.9999996, max abs
diff 1.8e-4 (6070). The GPU job repeats this check per encoder (48 genes) and stops on a cosine
below 0.999. A CPU dry run of the embed loop round-tripped 40 downstream windows through the h5
table; stored fp16 rows differ from the single-sequence `model.embed` by at most 0.0018.

### Verification (job 6090, `results/compute_isolate_composite_verify.json`)

Passed: 1,011 isolates, 6,015 reference genes each, every row of every table carries either an
input hash present in its sequence table or a typed gap, never neither and never both.
`embedding_complete` is false until job 6106 runs. Window geometry: of S288C-allele rows with a
window, 241,048 of 853,968 upstream windows (28.2%) and 474,437 of 857,646 downstream windows
(55.3%) are byte-identical to the S288C reference window (an off-by-one would make this 0). In a
sample of 5,000 non-identical upstream windows the median Hamming distance is 5; 23% exceed 500,
and of 300 of those 70% are within edit distance 20 (a flank indel), 10% beyond 200.

### Findings that change how the composite is read

- **The S288C CaLM table is not one convention.** For 6,319 genes the stored string is the CDS
  truncated to 3,072 nt. For the 288 intron or noncontiguous genes the stored string is the
  unspliced gene span, but the vector is the spliced-CDS embedding for 266 of them (re-embedding
  the spliced CDS reproduces it at cosine > 0.9999) and the unspliced-span embedding for 22
  (8 mitochondrial Q genes and YAL001C, YGL178W, YGL251C, YGR296W, YHR077C, YIL177C, YJL225C,
  YKL157W, YLL066C, YLL067C, YML133C, YNL339C, YPL075W, YPL283C). Seeding keys each
  row by the string it actually embedded (`index/s288c_seed_calm.parquet`); the 22 unspliced rows
  are not reused for isolates, whose CaLM input is always the spliced CDS.
- **28 genes are anchored differently in the S288C tables.** The SGD R64-4-1 gene span is not
  the Peter header span (R64-1-1 annotation) for these, e.g. YBL072C 88523-89125 against
  88523-89440. For 25 the SGD CDS is a single interval inside the Peter span (the CDS and protein
  here follow the SGD model, cut from the allele); 3 are SGD intron genes. The S288C windows were
  anchored on the SGD span, the isolate windows on the Peter span located in the assembly, so
  for these genes they do not match the S288C rows (`sgd_model_agrees` is false).
- **Frame.** 206,604 reference-gene rows over 3,516 genes have a CDS length that is not a
  multiple of 3, and 202,087 translate with an internal stop. A few genes carry it in nearly every
  isolate (YMR071C 1,004, YBR073W 1,000, YLL059C 997 of 1,011) with the S288C allele in only 2 to
  17 isolates and the allele 1 or 2 bp longer than SGD. Hypothesis (untested): S288C-specific
  indels or homopolymer calls in the allele inference; comparing against
  `genesMatrix_Frameshift.tab.gz` is the test. Stored as released.
- **Heterozygosity.** 31% of reference-gene rows in a 40-gene smoke pass carry IUPAC codes;
  ProtT5 reads them as X or unk, CaLM as unk codons, FUDT windows come from the haploid assembly.

### Mismatches between the runner on main and the reference geometry

Runner: `experiments/embeddings/compute_isolate_embeddings.py` (main).

- Translation strips the trailing `*` (`protein.rstrip("*")`); every S288C ProtT5 input keeps it.
- Intron genes are translated from the unspliced span (the tarball holds the gene span, introns
  included), so 239 genes get wrong proteins and CDS.
- The locus is the best HSP's start and end, never checked against the query ends; a hit that
  stops short (AAC: 5 of 40 full length in the feasibility run) slices a window off the gene's
  real start. Here the HSP may stop at most 10 bp short and the CDS end is extrapolated; more is
  `partial_alignment`.
- Short flanks are truncated and embedded (and optionally replaced by the S288C window); here a
  short flank is a typed gap, never a truncated or substituted window.
- Its assembly member index (and the tier's `1011Assemblies.tar.gz.member_index.tsv`, 1,010
  lines) misses CRL, whose member is `GENOMES_ASSEMBLED/CRL_.re.fa`.
- The window geometry itself matches (1,000 bp + start codon; stop codon + 297 bp, both strands).

### Next

Release job 6106 when a card frees (`scontrol release 6106`); it ends with the verify stage,
which then reports `embedding_complete`.

## 2026.10.22 - Untokenizable inputs are a typed gap: ambiguous_sequence

### The failure

Job 6106 ran 4 h 08 min. FUDT upstream finished: 1,131,360 computed rows in
`alleles/fudt_species_upstream.h5` (14,889 s, 76.0 windows per s on one RTX 6000 Ada, fp32,
batch 64). The next encoder, CaLM, raised `KeyError: 'UAM'` in `calm/alphabet.py encode`: a codon
holding the IUPAC code M (A or C, a heterozygous site in a diploid isolate's inferred allele).
Resumes 6121 and 6122 hit the same error in 36 s.

### What each encoder does with a character outside its vocabulary (measured from the tokenizers)

| encoder | vocabulary | behavior on anything else |
|---|---|---|
| CaLM | the 64 ACGU codons | `encode` indexes `tok_to_idx` directly and raises (fails loudly; no fallback) |
| FUDT up and down (SpeciesLM) | the 4,096 ACGT 6-mers | the 6-mer becomes `[UNK]` (id 1) silently: a fallback inside the encoder |
| ProtT5 | 20 amino acids plus B, O, U, X, Z | the wrapper maps U, Z, O, B to X; `*` and J become `<unk>` silently |

Findings that follow:

- FUDT upstream in job 6106 embedded 111,641 distinct windows (138,408 isolate-gene rows) that
  contain N, an IUPAC code or soft-masked lowercase bases, each offending 6-mer read as
  `[UNK]`. Those h5 rows are now labelled `untokenizable` in the `source` column and nothing in
  the index points at them.
- Every ProtT5 input, the 6,607 S288C rows the composite trained on included, ends in `*`,
  which ProtT5 reads as `<unk>`. That is the trained convention and is kept: the protein
  alphabet is the 20 amino acids plus `*`.
- A protein with many internal stops tokenizes SHORTER than one token per residue (in 20,000
  tokenizable proteins, 461 were shorter by 1 to 30): sentencepiece merges runs around `*`.
  Job 6128 bucketed by `len + 1`, formed a padded batch and the no-padding assertion stopped
  it; batches now bucket by the tokenizer's own count (commit 4fd4a4f98).
- An ambiguous codon translates to X (or B, Z, J for two-way ambiguities); a codon whose
  ambiguity is synonymous translates to its amino acid, so a protein can be clean where the
  CaLM input is not.
- The lowercase a, c, g, t, n codes are soft-masked assembly bases, not ambiguities. Under the
  rule they are gaps (about 3% of the distinct untokenizable windows). Hypothesis (untested): uppercasing
  at slice time would recover them; that is a change of the stored window string and needs the
  owner's call and an align plus merge rerun (both a few minutes).

### The rule

An input with any character outside its encoder's alphabet (`ENCODER_ALPHABET`: ACGT for CaLM
and both FUDT windows; the 20 amino acids plus `*` for ProtT5) is a typed gap
`ambiguous_sequence`: never resolved to a base, never embedded. `--stage alphabet` (job 6126,
1:26 wall, 9.1 GB) writes it into `index/isolate_gene.parquet` per encoder (`protein_gap`,
`calm_gap`, `up_gap`, `down_gap`; protein and CaLM no longer share `cds_gap`), with the
offending codes (`<kind>_untokenizable_codes`) and the hash moved to `<col>_untokenizable`, so
a row still holds a key XOR a gap; lists the inputs in `sequences/untokenizable_<kind>.parquet`;
and records the counts in the manifest (`representation_gaps`) and
`results/compute_isolate_composite_alphabet.json`. `--stage embed` skips and counts such inputs
before any batch, and `Embedder.__call__` raises if one reaches the model. No S288C reference
input is affected.

### Counts (job 6126)

| encoder | distinct inputs | distinct untokenizable | isolate-gene rows untokenizable | of rows with an input | codes (rows) |
|---|---|---|---|---|---|
| FUDT upstream | 1,137,931 | 111,641 | 138,408 | 2.4% | N 128,500; lowercase a/c/g/t/n 2,035 to 3,601 each; R 2,063, Y 2,057, W 626, K 455, M 447, S 337 |
| FUDT downstream | 413,606 | 48,211 | 71,495 | 1.2% | N 65,872; lowercase 1,106 to 2,399 each; Y 672, R 657, W 283, S 186, K 170, M 141 |
| CaLM | 1,163,337 | 580,786 | 1,281,220 | 20.8% | R 906,282, Y 905,712, W 316,113, M 308,503, S 287,950, K 265,003, N 654 |
| ProtT5 | 757,720 | 384,755 | 992,159 | 16.1% | X 966,332, B 101,169, J 33,309, Z 28,703 |

All 1,011 isolates are touched for every encoder. A row can carry several codes, so the code
counts overlap.

### Jobs (two cards, both resumable)

| job | what | id | measured |
|---|---|---|---|
| A | `--encoders fudt_species_downstream,calm`, 4 h | 6127 | downstream 358,838 to compute at 265 to 267 per s (about 23 min); CaLM 576,007 to compute at about 285 per s on the shortest CDSs, about 49,000 codon tokens per s |
| A resume | same line, afterany 6127, 4 h | 6131 | |
| B | `--encoders prot_t5_xl_uniref50` | 6128 failed (padded batch), 6138, 6 h | 366,429 to compute; about 75 per s on proteins under 311 aa, about 24,500 tokens per s |
| B resume | same line, afterany 6138, 4 h | 6132 failed with 6128, 6139 | |
| verify | CPU, afterany 6131 and 6139 (reads every table, so it waits for both writers) | 6135 | |
| representation | build 6129 (done), segregant pass 6130, verify 6133, trainer prepare 6134 | | |
| smokes | `gh_train_043_smoke.slurm` (6124's line) 6136; with `SAMPLER=balanced` 6137; afterany 6131 and 6139, afterok 6134 | | |

Time projections (hypothesis from token mass, linear in tokens; attention makes the long tail
slower): CaLM 3.17e8 codon tokens left after the first 45,329 inputs, about 1.8 h; ProtT5 2.16e8
residue tokens left after the first 77,522, about 2.4 h.

### Outcome (2026-10-22 evening)

- Job A 6127 COMPLETED in 3:12:40: FUDT downstream 358,838 rows, CaLM 576,007 rows in
  10,213 s (56.4 per s overall). Resume 6131 found nothing left (36 s).
- Job B 6138 COMPLETED in 5:41:02: ProtT5 366,429 rows in 20,454 s (17.9 per s overall).
  Resume 6139 found nothing left (30 s).
- Correction: the token-mass projections above (CaLM about 1.8 h, ProtT5 about 2.4 h) were
  low; the long inputs at the end ran far slower per token than the short ones measured first.
- Verify 6135 (1:44) PASSED with `embedding_complete: true`: every row with an input has an
  embedding row (FUDT upstream 5,570,717, FUDT downstream 5,685,119, CaLM 4,882,463, ProtT5
  5,171,524) and every other row carries a typed gap (`results/compute_isolate_composite_verify.json`).
- Smoke 6136 passed the trainer's key check and is training (0.035 s per step, 3.7 min per
  epoch of training by its 30-step probe); 6137 (balanced) queues behind it.
