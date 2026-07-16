---
id: tx5rd5daoawlwgymdpocf0r
title: Expression Modeling Setup
desc: ''
updated: 1784091583519
created: 1784091583519
---

Future-modeling design for the paper section **"Natural Genetic Variation vs
Model-Design Perturbations"** (Fig. 4), spun out of the 018 bit-accounting analysis
([[experiments.018-natural-isolate-genomics]]). This note is the *plan*, not a built
experiment. Parent finding, in one line: a KO genotype is ~14.7 bits (one gene index),
a natural isolate's is ~3.3 Mbit of real sequence divergence, and both drive a
~180 kbit expression response — so natural variation is the **only** modality where a
sequence encoder has anything to read.

## Datasets (so this is findable later)

Everything reduces to a **log2 expression ratio vs a reference**, node-level over the
~6,000 S288C ORFs.

| dataset | loader | platform | strains | genotype | label |
|---|---|---|---|---|---|
| **Kemmeren 2014** | `MicroarrayKemmeren2014Dataset` | microarray | 1,484 **single KO** | 1 gene deleted | `expression_log2_ratio` = log2(mut / WT-refpool) |
| **Sameith 2015** | `Sm/DmMicroarraySameith2015Dataset` | microarray | 82 single + 72 **double KO** | 1–2 genes deleted | same |
| **Caudal 2024** | `CaudalPanTranscriptome2024Dataset` | RNA-seq | 943 **natural isolates** | ~5,000 sequence variants + presence/absence | `expression_tpm` → log2(iso / population-mean) |

## 2026.07.14 - The question, and how to actually pose it

**The ultimate goal** is not a benchmark number — it is to show that combining
YKO-library expression and natural-isolate expression yields a **better representation
of the cell**, and, through Fig. 1c, to *motivate encoding sequence at all*. The Fig. 1c
bridge is persistent entities (genome/sequence) ↔ contingent observations (experiments).
If you only ever knock genes out, the sequence never changes across strains, so the
persistent-entity side is a constant and the bridge is a trivial lookup — there is no
reason to encode. Natural variation is what makes the persistent-entity side *vary per
observation*, which is the empirical motivation we want to supply.

**The reframe that dissolves most of the uncertainty:** this is TWO questions, not one.

1. *Data-augmentation:* does adding natural-isolate data improve prediction? (and the
   reverse — does KO data improve natural-isolate prediction?)
2. *Representation:* does sequence encoding earn its keep, and **specifically because
   sequence varies**? This is the Fig. 1c question, and it is the sharper one.

### The design fact that settles the motivation before any run

A natural isolate's genotype is **unrepresentable** in a deletion-only model. The CGT's
perturbation vocabulary for a KO is "which gene nodes are removed" — a set of indices. A
natural isolate deletes ~zero genes; it *edits* thousands (018: median 59 genes respond
vs 4 for a single KO). The only way to encode "gene X carries a different allele" is to
change gene X's **node features** — i.e. re-embed its sequence. So the sequence encoder
is not a performance add-on; it is the **precondition** for ingesting natural variation
at all. That is a fact about the data model, not an empirical result to win.

### Fitness is the wrong label (why)

Single-KO fitness is near-degenerate: the distribution is dominated by ≈WT values, so a
model predicting "≈1 for everything" scores well and gets its correlation almost entirely
from the essential-gene tail. It can be hit from coarse features that never touch
sequence. The Fig. 7 traditional-ML SI result ("fitness is easy; interactions are hard")
is the *foil* that justifies retiring fitness — not a reason to make interactions the
target. **Expression reconstruction** (the ~6,000-dim vector) is the right task: it is
high-dimensional and forces the model to use gene-specific, hence sequence-specific,
information.

## 2026.07.14 - Clearing up three things that were confusing

### 1. "gene-ID embedding vs sequence embedding" — what the ablation actually is

- **gene-ID embedding** = a learnable lookup table, one vector per gene systematic name,
  **constant across strains**. Same dimensionality as the sequence embedding for a fair
  capacity comparison.
- **sequence embedding** = ESM2 / nucleotide-transformer / fungal-up-down embedding of
  the gene's **actual sequence in that strain**.

The worry "the learnable ID will just do whatever is necessary, so it might not do worse"
is exactly what the ablation tests — and there are **two reasons it must lose on natural
isolates specifically**:

- **Alleles.** A learnable ID is one vector per gene, identical for every strain, so it
  *cannot* distinguish isolate A's allele of gene X from isolate B's. It is blind to the
  variation that drives the isolate's response. On **KO data** the ID can tie with the
  sequence embedding (the sequence is constant — there is nothing extra to exploit), and
  that tie is the point: encoding earns nothing where sequence does not vary.
- **Accessory genes.** Natural isolates carry ORFs absent from S288C. A fixed-vocabulary
  ID embedding has **no slot** for an out-of-reference ORF — it literally cannot represent
  it. A sequence embedding handles it natively (embed whatever sequence is present). KO
  data never has this problem, so again the failure is specific to natural variation.

So the expected 2×2 is a clean interaction: sequence ≈ ID on KO-only; sequence ≫ ID once
natural isolates are in play.

### 2. "genotype magnitude, r = 0.04" — what that even meant

Confusing phrasing on my part in 018. It is **not** the expression vector. For each
isolate I computed a **scalar summary of how far its GENOME is from S288C** — (a)
genome-wide SNP divergence (% of bases differing), and (b) natural-KO burden (# reference
ORFs absent/broken) — and correlated that scalar against the **number of differentially
expressed genes**. `r = 0.04` means: an isolate whose genome is twice as diverged does
**not** differentially express more genes. Renamed here to **"genome-wide divergence from
S288C (scalar)"** to kill the ambiguity.

**Modeling consequence:** the map from genome-divergence-size → response-size is weak, so
predicting *how much* an isolate's transcriptome moves will be hard. The winnable signal
is *which* genes move and in *what direction* (through the graph), not the magnitude. Score
accordingly — per-gene direction/rank agreement, not just magnitude MSE — or the isolates
will look harder to predict than the mechanism claim deserves. ("isolates look harder" =
a magnitude-only metric will report low performance and be mistaken for the model failing,
when it is the metric probing the one thing that genuinely is not there.)

### 3. "matched target values" — a shared encoder with two decoder heads

Your read is right: matched-target ⇒ one decoder; the alternative ⇒ two decoders.

- **Option A — one decoder, matched targets.** Force Kemmeren and Caudal expression into
  one output space. Requires them comparable, which they are not: microarray log2(mut/WT)
  vs RNA-seq log2(iso/pop-mean), platform confounded with modality. Batch effect
  contaminates the result.
- **Option B — shared encoder, two decoder heads (recommended).** The **encoder**
  (genotype → latent) is shared across both modalities; each modality gets its **own
  decoder head** (latent → expression), so the platforms never have to share an output
  scale. Only the *representation* is shared. This is what "let isolates contribute
  through the shared encoder, not through matched targets" means — and it is what cleanly
  enables the bidirectional test below.

## 2026.07.14 - The experiments (updated plan)

Task everywhere: reconstruct a held-out strain's expression vector. Encoder = the
**Cell Graph Transformer (CGT)** (`equivariant_cell_graph_transformer`; "CGT" for short).
Gene features from `esm2` / `protT5` / `nucleotide_transformer` / `fungal_up_down_transformer`
(all already in torchcell). The 2×2 is a **config sweep, not new modeling**.

- **E1 — representation ablation (the headline).** 2×2: {ID embedding, sequence embedding}
  × {train KO-only, train KO+natural}. Test = held-out **KO** strains. The **interaction**
  is the story (sequence ≈ ID on KO-only; the lift appears only with natural isolates + a
  sequence encoder). The `ID + natural` cell is the tell — weak or ill-defined because ID
  cannot encode alleles or accessory ORFs.
- **E2 — cross-modality transfer, BIDIRECTIONAL (the strongest single result).** Train on
  natural isolates → test on held-out KOs, **and** train on KOs → test on held-out natural
  isolates. Transfer in either direction is direct evidence the model learned a *shared*
  genotype→expression mechanism rather than memorizing strains. Both directions matter:
  "does natural variation help cell modeling" AND "can the KO library help reconstruct
  natural-isolate expression" — the two halves of "combine the data for a better cell
  representation."
- **E3 — held-out genes (harder generalization).** Predict expression at genes never
  perturbed in training.
- **E4 — digenic (Sameith doubles), a network-perturbation test.** Interesting precisely
  because a double deletion perturbs a *network*, not a node. **Gated on #72** (see below)
  — its labels are currently 24% sign-corrupted, so "doubles are hard" would partly be
  measuring our own loader bug.
- **Control (essential).** KO+natural vs KO + a matched-size augmentation that carries no
  new sequence signal (genotype-shuffled isolates, or the ID-embedding arm). Isolates the
  *sequence information*, so the claim is "natural variation helps," not "more rows help."
- **Foil.** Show single-KO fitness is near-degenerate (retire it as a label), using the
  Fig. 7 SI result.

## 2026.07.14 - Blocking caveats (fix before building)

- **#71 — Caudal missing gene-absence edits.** Re-verified against `main` today: **133
  absences/isolate (134,428 total) silently dropped**; the absence loop guards on
  `core_mask` (frequency) instead of reference membership, and `s288c_mask` is passed in
  but unused. The natural-isolate *genotype input* is therefore incomplete — fix + rebuild
  the Caudal LMDB before it is a model input, or the encoder conditions on a corrupted
  genotype.
- **#72 — Sameith double-KO sign.** Re-verified today: the **global** sign convention is
  correct (do not flip it), but **70/287 arrays (24%)** are per-array backwards because
  GEO declares both `#VALUE` orientations within GSE42536 and the loader assumes one. The
  double-KO *labels* are partly corrupted — fix before E4.
- **Cross-platform batch effect.** Handled by Option B (two decoder heads); do not force a
  matched target space. Keep targets per-gene standardized *within* modality.

## Open decisions for the author

- Primary metric: per-gene **direction/rank agreement** vs magnitude MSE (recommend the
  former, given r = 0.04).
- Whether E4 (digenic) is in scope for Fig. 4 or held for a follow-up — it is the most
  biologically interesting (network perturbation) but is gated on the #72 rebuild.
- Split granularity for E1/E2: held-out **strains** (predict a new perturbed strain) is
  the primary; add held-out **genes** (E3) for the harder claim.

## 2026.07.16 - Fig 4 panels: one plot, one question

Refined the figure into a **panel-per-question** layout. Descriptive panels (a–f) are
buildable now from 018 data; modeling panels (g–h) are the E-series above. The
descriptive set lives in a paired note + script,
[[experiments.018-natural-isolate-genomics.dataset-comparison]] (`dataset_comparison.py`).

| # | Question | Panel | Status |
|---|---|---|---|
| a | What are we comparing? | setup schematic (KO → double KO → natural isolate on a genotype-edit axis) | author draw.io |
| b | **How genetically different are these strains?** | # reference ORFs absent (x) vs % sequence divergence on shared genes (y); KO at (1, 0), isolates far out | ✅ built |
| c | genotype design-space coverage | folded into b (the two axes *are* the coverage) | b |
| d | **What do their transcriptomes look like?** | Kemmeren-single / Sameith-single / Sameith-double / Caudal as **matched spread bands** on one scale | ✅ built |
| e | How many genes move? | per-strain DE-count distribution, KO vs natural (single hue per dataset) | ✅ built |
| f | transcriptome design-space coverage | **PCA + UMAP** of the joint expression matrix, coloured by dataset | ✅ built |
| g | Does natural variation improve KO prediction? | E1 2×2 interaction | ⛔ modeling |
| h | Can each modality reconstruct the other? | E2 bidirectional transfer | ⛔ modeling |

**PCA/UMAP methodology (panel f).** Each strain = its log2 expression vector on the
shared S288C ORFs; stack all datasets, **z-score per gene within each dataset**, restrict
to shared measured genes, then PCA (linear, interpretable, global geometry) **and** UMAP
(non-linear, cluster/coverage structure). Caveat stated on the panel: Kemmeren/Sameith are
microarray log2(mut/WT) and Caudal is RNA-seq log2(iso/pop-mean), so some separation is
**platform, not biology** — the within-dataset z-score is the mitigation, and it is the
same batch confound the modeling side dodges with two decoder heads (Option B).

**E4 (digenic) is now unblocked** — #72 landed (Sameith per-array dye-orientation fixed),
so the double-KO labels are corrected. Sameith doubles enter panel d as their own band and
are in scope for E4.

**Bit ledger — retracted from Fig 4.** The per-strain *phenotype* codelengths come out ~equal
across all three datasets only because each is a ~6,000-float vector and the gzip cost is
~80–90% serialization (repeated gene-name keys + float-as-ASCII), not biological signal — so
that "equality" is an artifact, not a result. The real, keep-able finding is the *genotype*
codelength: a KO encodes in ~15 bits, a natural isolate in ~3.3 Mbit — the isolate genome
**requires more information because it contains more variation**. That belongs with the
**supported-datasets table** (`Signal (gzip)` column), not a Fig 4 panel and not Fig 1c.
