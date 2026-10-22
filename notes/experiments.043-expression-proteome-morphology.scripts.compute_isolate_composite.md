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
