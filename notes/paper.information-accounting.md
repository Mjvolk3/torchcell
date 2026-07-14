---
id: 779c1j10b1w26li1irxuyus
title: Information Accounting
desc: ''
updated: 1782239925374
created: 1782239925374
---

Companion math note for the Nature Biotechnology CGT paper
([[paper.nature-biotech-cgt-outline]]). Justifies the model-construction strategy
— **why we look up / pretrain entity representations and fit phenotype labels only
on a thin perturbation operator** — with a simple information accounting.

> **Canonical bit estimates for Fig 1c are the `2026.07.12` section at the bottom**:
> **persistent entities ${\sim}10^{13}$ vs contingent observations ${\sim}10^{10}$ compressed
> bits**, *measured* from the public archives by
> [[experiments.016-information-accounting.scripts.persistent_entity_corpus_sizes]].
> This **supersedes** the `2026.07.03` section's ${\sim}10^{10}$ vs ${\sim}10^{8}$, which was a
> hand-derived bound (fungal genomes × 2 bits/bp; measurements × ~10 bits) and is kept below only
> for history. The material in `2026.06.23` (per-genome budgets, the core inequality, corollaries)
> is still the fuller *identifiability* derivation and is **not** superseded — but note that it
> reasons about **effective supervision bits** ($10^6$–$10^7$), which is a *different quantity*
> from the compressed corpus size, and the two must not be conflated.

Final figure exports live in [[paper.nature-biotech.figures]].

## 2026.06.23 - Information accounting: why the model cuts between entities and instances

**Status.** Symbolic argument is complete; empirical constants tagged `[PIN]` need
one genomics pass (sequence variation) and one assay-noise pass (label bits) to
fill. Every conclusion here is an **order-of-magnitude** statement and is robust to
±1 OOM in any single constant — the gap we exploit is 2–4 OOM wide.

**One-sentence thesis.** Sequence/structure data (the *universe of things*) is
abundant; phenotype data (the *universe of instances* — perturbed, measured cells)
is scarce by 2–4 orders of magnitude. The only **identifiable** place to cut a
genotype→phenotype model is therefore *between* them: learn or look up entity
representations on the side where data is plentiful, and spend the scarce phenotype
labels **only** on the thin perturbation operator that maps entities → measured
cells.

### Setup — notation and the cut

One notation, consistent with the model note
([[torchcell.models.equivariant_cell_graph_transformer]]) and the slide deck.
**No tildes:** here $G$ is *always* the cell graph, so the graph-vs-cell-graph
disambiguation the tilde bought is unneeded, and $\tilde{(\cdot)}$ stays reserved
for the row-normalized adjacency $\tilde A$ (model note). Calligraphic = space,
plain = element.

| symbol | meaning |
| --- | --- |
| $G$ | wildtype **cell graph** — genome sequence + gene/regulatory/PPI networks + metabolism, one multi-graph with vertex/edge features |
| $V(G),\,g_i,\,N$ | genes = vertices; gene $g_i$; $N=\lvert V(G)\rvert=6607$ |
| $\varepsilon$ | **environment / context** vector |
| $S\subseteq V(G)$ | **perturbation** — perturbed gene set, type $\tau\in\{\mathrm{del},\mathrm{OE},\mathrm{KD}\}$ |
| $\mathrm{Embed}$ | **entity encoder** $g_i\mapsto x_i\in\mathbb R^{d}$ — a *pretrained* model (ESM / DNA-LM / chem) **or** a *learnable lookup* $E\in\mathbb R^{N\times d}$ |
| $F_\theta$ | **encoder**: $H=F_\theta(G)$, transformer over the embedded cell graph |
| $H=(h_{\mathrm{CLS}},h_1,\dots,h_N)$ | **cell representation** (hidden); $h_{\mathrm{CLS}}$ whole-cell, $h_i$ per-gene, $\in\mathbb R^{d}$ |
| $\mathcal T_\psi$ | **perturbation operator**: $H_{\mathrm{pert}}=\mathcal T_\psi(H,S)$, equivariant, $\sim5\times10^4$ params |
| $\mathcal R_\phi$ | **readout / decoder**: multitask heads → fitness, interaction, expression, morphology |
| $y,\hat y\in\mathcal Y$ | observed / predicted **phenotype** |
| $\mathcal L,\,D,\,\theta,\psi,\phi$ | loss; data distribution over $(G,\varepsilon,S,y)$; encoder / operator / readout parameters |
| $\mathcal U,\,\mathcal I$ | **universe of things** (entities) / **universe of instances** (the contingent, $=\operatorname{supp}D$) |

The model is **encode → operate → decode**, and the whole argument turns on the one
**cut at the representation $H$**:

$$
\underbrace{G}_{\substack{\text{entities}\\ \text{things }\mathcal U}}
\;\xrightarrow{\;F_\theta\;}\;
\underbrace{H}_{\substack{\text{representation}\\ h_{\mathrm{CLS}},\,h_i}}
\;\xrightarrow{\;\mathcal T_\psi(\,\cdot\,,\,S)\;}\;
\underbrace{H_{\mathrm{pert}}}_{\text{perturbed}}
\;\xrightarrow{\;\mathcal R_\phi(\,\cdot\,,\,\varepsilon)\;}\;
\underbrace{\hat y}_{\substack{\text{phenotype}\\ \text{instances }\mathcal I}}
\qquad
\hat\theta,\hat\psi,\hat\phi=\arg\min_{\theta,\psi,\phi}\ \mathbb E_{(G,\varepsilon,S,y)\sim D}\big[\mathcal L(\hat y,y)\big]
$$

A cell is a **set of entities** $\{e_1,\dots,e_k\}\subset\mathcal U$ assembled into
$G$ and embedded by $\mathrm{Embed}$ — *instances are composed from things* (what
Figure d/e draws, below). This restates the paper's framing — *"strain =
perturbation operator over a shared wildtype reference"* — information-theoretically.

**The cut sits at $H$.** Encoding ($F_\theta$, including $\mathrm{Embed}$) is paid
for by *entity/sequence* data — pretraining, or a free lookup. Operating + decoding
($\mathcal T_\psi,\mathcal R_\phi$) is the *only* part fit on phenotype labels. The
$\mathcal U\,\vert\,\mathcal I$ boundary is exactly this cut — and the claim below is
that it is forced by the data, not chosen for elegance.

### Three budgets (count the bits on each side)

Define three quantities and then compare them. Real in-repo counts are stated;
unknowns are tagged `[PIN]`.

#### Budget A — capacity to *represent* the entities (the things side)

Bits a representation must carry to distinguish the entities seen in data. Three
ways to count, increasingly tight:

- **A1 — raw genome ceiling.** Yeast genome $L\approx1.2\times10^7$ bp at 2 bits/bp
  $\Rightarrow H_{\max}=2L\approx 2.4\times10^7$ bits $\approx 3$ MB. "If every base
  could matter."
- **A2 — functional content.** Most bases are not independently phenotype-relevant.
  Coding sequence is $\sim$70% of the genome across $\sim$6,600 genes; effective
  per-residue entropy after conservation is well under the 4.2-bit amino-acid
  ceiling. Defensible mid-estimate $\sim 10^6$–$10^7$ bits/genome.
  `[PIN: tighten with an MSA/conservation (e.g. per-column entropy) estimate.]`
- **A3 — variation across our strain panel (the bits that actually change
  in-distribution).** This is the number a model must encode to *tell our strains
  apart*. For a panel of $S$ related *cerevisiae* genomes segregating at $V$
  variable sites with mean per-site allelic entropy $\bar h$,
  $$H(\text{genome}\mid\text{panel}) \approx V\cdot\bar h.$$
  At 1011-genomes scale, $V\sim10^5$–$10^6$ segregating sites, $\bar h\lesssim 1$ bit
  $\Rightarrow \sim 10^5$–$10^6$ bits of *natural* sequence variation.
  `[PIN: recompute V and h̄ from the specific related-genome papers we hold.]`

So the *things* side is $10^5$ (natural variation) to $10^7$ (functional content)
bits **per genome**, and the universe of things *as a whole* (all proteins / RNAs /
metabolites, across organisms) is effectively unbounded — which is precisely why it
has its own giant pretraining corpora.

#### Budget B — supervision available from phenotype labels (the instances side)

How many bits the measurements actually carry. Two ways:

- **B1 — naive label bits.** $N$ measurements $\times\ b$ effective bits each. In
  repo (Costanzo 2016 + Kuzmin 2018, see
  [[torchcell.knowledge_graphs.create_scerevisiae_kg_small]]):
  $N_{\mathrm{SMF}}\!\approx\!2.2\times10^4$, $N_{\mathrm{DMF}}\!\approx\!1.4\times10^6$,
  $N_{\mathrm{TMF}}\!\approx\!9.1\times10^4$ → $N\approx1.5\times10^6$ fitness/
  interaction values. A noisy real-valued assay resolves
  $b\approx\log_2(\sigma_{\text{signal}}/\sigma_{\text{noise}})\approx 2$–$4$ bits.
  $$B_1\approx 1.5\times10^6\times 3 \approx 5\times10^6\ \text{bits}\ (\approx 0.6\ \text{MB}).$$
  `[PIN: set b per assay from its measured CV / replicate noise.]` Expression and
  morphology add raw bits (each transcriptome/CalMorph readout is high-dimensional),
  but both are strongly low-rank — see B2.
- **B2 — effective (non-redundant) label bits.** Genetic-interaction matrices are
  approximately **low rank**: signal lives in $r$ latent factors ($r$ in the tens–
  hundreds), not in $N$ independent numbers. Independent supervision is closer to
  $$B_2 \approx r\cdot G\cdot b \approx 10^2\cdot 6.6\times10^3\cdot 3 \approx 2\times10^6\ \text{bits},$$
  and likely less. **The usable supervision sits at the low end of $10^6$–$10^7$.**
  `[PIN: estimate r from the SVD/PCA of the interaction matrix.]`

Crucially this total is **shared** across *all* entity-representation parameters
**and** *all* perturbation-operator parameters — every bit you spend representing an
entity is a bit you cannot spend on the genotype→phenotype map.

#### Budget C — capacity of an *end-to-end* sequence model (what end-to-end forces the labels to fit)

A model mapping raw sequence → phenotype has $P$ parameters. Real sequence
foundation models are $P\approx10^8$–$10^{10}$ (ESM2 8M→15B; DNA-LMs 0.5B→7B); even
a modest from-scratch encoder is $\gtrsim 10^7$. These are trained on sequence
corpora of $10^{10}$–$10^{12}$ tokens — i.e. the things side **is** data-rich, but
with **sequence** data, not phenotype data.

### The core inequality (the whole argument in one line)

Treat parameters as the bits of capacity that supervision must pin down (the MDL
two-part-code / effective-dimension heuristic — see *Caveats*; the 2–4 OOM gap below
dwarfs the slop in this heuristic). Identifiability requires roughly

$$
\underbrace{B}_{\text{usable label bits}\ \sim\,10^6\text{–}10^7}
\;\gtrsim\;
\underbrace{C_{\text{fit}}}_{\text{capacity the labels must fit}} .
$$

- **End-to-end:** $C_{\text{fit}}=C\sim10^8$–$10^{10}$. The inequality is **violated
  by 2–4 OOM** — you estimate $10^8$–$10^{10}$ parameters from $\le 10^7$ bits of
  supervision, underdetermined by $\sim10^2$–$10^4\times$. The phenotype data cannot
  pay for the entity representation. (Consistent with Ahlmann-Eltze et al., *Nat
  Methods* 2025: perturbation transformers fail to beat linear baselines — too few
  labels for the capacity.)
- **Factored (ours):** move the entity-representation cost **off** the phenotype
  budget. The encoder $F_\theta$ (with $\mathrm{Embed}$) is paid for by *sequence*
  data (pretraining) or by a *free* lookup whose rows the labels pin only up to
  dimension $d$; the labels then fit **only** the operator + readout
  $\mathcal T_\psi,\mathcal R_\phi$, whose capacity is small by construction
  ($C_{\text{fit}}=|\psi|+|\phi|\sim5\times10^4 \ll B$). **The inequality is restored.**

This is the precise statement of *put the parameters where the data is.* The
things/instances boundary is the **unique** cut that places every data-hungry
component on the side with a matching data source.

### Corollary 1 — lookup $\ge$ sequence embedding when sequence is fixed (explains our result)

When no sequence changes across instances (deletions of existing genes, environment
shifts), the entity that varies is a **categorical label** over the $N\!\approx\!6{,}607$
genes. The sufficient statistic for any function of gene identity is a free vector
per gene — exactly a learnable lookup $x_i=E[g_i]$. A pretrained sequence embedding
$\mathrm{Embed}_{\mathrm{seq}}(g_i)$ is then a *fixed, possibly-misaligned reparameterization*
of the same categorical; it can only inject an inductive bias, and a misaligned bias
**hurts**. Formally the extra information a sequence model adds is

$$
I(\text{seq};\,y \mid \text{gene id}) = 0 \quad\text{when sequence is constant given identity.}
$$

Hence for fixed-sequence fitness, **lookup $\ge$ sequence embedding** — matching the
in-repo results: the learnable-embedding fitness runs ([[scratch.2025.01.16.143744]],
val Pearson $\approx0.58$ with `learnable_embedding`, vs sequence-derived
`codon_frequency` near zero) and the classical-ML embedding sweep behind Suppl.
Fig. S1 ([[experiments.smf-dmf-tmf-001.results]], [[experiments.002-dmi-tmi]]).

### Corollary 2 — sequence-function *required* when sequence varies (your hypothesis, formalized)

When perturbations include **sequence changes** (point mutants, engineered alleles,
heterologous genes, cross-organism transfer), the varying object lives in sequence
space $\Sigma^L$, of cardinality $|\Sigma|^L$ — astronomically larger than any
lookup can enumerate. A lookup has **no row** for an unseen variant; its
generalization error there is unbounded. Only a smooth $\mathrm{Embed}(\cdot)$ over sequence
space can place an unseen variant near its neighbors and transfer phenotype:

$$
\text{lookup sufficient} \iff \text{variation is a finite enumerable set;}\qquad
\text{sequence-}\mathrm{Embed} \text{ required} \iff \text{variation has support in } \Sigma^L .
$$

This **is** your construction rule —
> *if a class of entity varies in the dataset and that class has phenotypic effect,
> instantiate that object (with a sequence-aware encoder) in the model* —

restated precisely: include the sequence encoder $\mathrm{Embed}_m$ for entity-class
$m$ iff $\Pr[\text{class-}m\text{ varies}]>0$ **and** $I(\,\text{class-}m;\,y\mid \text{rest})>0$. The
hypothesis *"sequence embeddings only help when sequence changes"* is the
contrapositive of Corollary 1 and is **directly testable**: hold out
engineered/variant strains and measure $\Delta = $ perf(sequence-$\mathrm{Embed}$) $-$
perf(lookup). Predict $\Delta\approx0$ on fixed-sequence tasks, $\Delta\gg0$ on
variant tasks. *(This is the one experiment that turns the hypothesis into a result.)*

### Corollary 3 — many encoders, one merge point (the multi-modal things stack)

The universe of things spans protein, RNA, DNA, small molecules, networks — each
with a mature pretrained model and its own data-rich corpus. Fine-tuning all of them
end-to-end against $\le10^7$ phenotype bits is not just burdensome, it is
**anti-identifiable** (Budget C, summed over modalities). The information-correct
design: train each modality's encoder where *its* data lives, and **merge them only
at the instance boundary** — the perturbation operator consumes the frozen/lookup
representation $H$ and is the sole module fit on phenotype.

This is the **Platonic Representation Hypothesis** (Huh, Cheung, Wang, Isola, ICML
2024): independently-trained large encoders converge toward a shared statistical
model of the world. The perturbation operator is the thin, data-cheap map from that
shared representation of things to the contingent measured cell. *(This is the
academic name for the figure's "pretrained model = universe compression / glimpse
the world of forms." Keep the **idea**; drop the platonic vocabulary in the figure.)*

### Corollary 4 — the evolutionary-coverage caveat (limit of pretraining)

Pretraining $g$ generalizes only over the manifold its corpus covers. Natural
sequence databases densely sample the **evolutionarily accessible manifold**
(homologs along selective paths), so $g$ interpolates well for natural variants and
for organisms close in evolutionary space — *"cover enough nearby organisms and we
have a chance."* But **engineered DNA** (synthetic constructs, non-natural edits,
designed pathways) lies *off* that manifold; evolution never sampled it, so even a
strong $g$ extrapolates poorly there. Cross-organism data extends the things-side
representation **along the natural manifold** but does **not** automatically solve
the engineered-sequence case — a separate data/representation frontier, and an
honest limitation to state in the Discussion.

### When does end-to-end become viable? (a concrete data target)

The factorization is forced by today's $B\ll C$. As phenotype data unify across the
ontology, $B$ grows; end-to-end becomes admissible when $B\gtrsim C_{\text{fit}}$,
i.e. (matched bit units)

$$
N_{\text{eff}}\cdot b \;\gtrsim\; P_{\text{seq-encoder}} .
$$

With $P_{\text{seq}}\sim10^8$ and $b\sim3$, that needs $N_{\text{eff}}\sim3\times10^7$
**independent** measurements — $\gtrsim20\times$ our current *effective* supervision
(and far more once low-rank redundancy is counted). This is a falsifiable threshold:
it says *when* sequence→phenotype end-to-end becomes the right architecture, and it
frames the data-unification effort quantitatively. Until then, the cut stays at the
things/instances boundary.

### Placeholders to pin (computation plan)

| Symbol | Meaning | How to fill | Source |
| --- | --- | --- | --- |
| $V,\ \bar h$ | segregating sites + per-site entropy across our strain panel | VCF/alignment of the related *cerevisiae* genomes → count variable sites, compute allele-frequency entropy | the genome papers we hold `[PIN: list them]` |
| A2 bits | functional sequence content per genome | per-column entropy over MSAs of the 6,607 proteins (or a conservation proxy) | proteome + an MSA/conservation tool |
| $b$ | effective bits per phenotype measurement | $\log_2(\sigma_{\text{signal}}/\sigma_{\text{noise}})$ from replicate CV, per assay | Costanzo/Kuzmin replicate stats; expression/morphology noise |
| $r$ | effective rank of the interaction signal | SVD/PCA of the gene×gene (and trigenic) interaction matrix; count factors to ~90% variance | in-repo interaction data |
| $P_{\text{seq}}$ | capacity of a candidate end-to-end encoder | parameter count of the smallest credible sequence→phenotype model | ESM2 / DNA-LM sizes |

The argument's **shape** does not depend on these; they sharpen the headline ratios
$B/A$ and $B/C$ from "2–4 OOM" to specific numbers for the Methods/Discussion.

### Caveats / where this is a heuristic, not a theorem

- **"1 param ≈ 1 bit" and "labels ≳ params"** are the MDL/effective-dimension
  heuristic, not a tight generalization bound. The rigorous versions
  (VC/Rademacher, PAC-Bayes, or an explicit two-part code) carry constants and
  log-factors. We rely only on the **order of magnitude**, where a $10^2$–$10^4$ gap
  swamps those constants.
- **Low rank of interactions** (B2) is an empirical regularity to confirm by SVD,
  not an axiom; if rank is higher, $B$ rises but stays $\ll C$.
- **$I(\text{seq};y\mid\text{id})=0$** holds for *strictly* fixed sequence; tiny
  residual variation (strain-background SNPs) makes it small-but-nonzero, which only
  sharpens the testable $\Delta$ in Corollary 2.

### Related

- [[paper.nature-biotech-cgt-outline]] — paper spine; this note backs Intro beat 2
  and the Discussion's "single organism today → cross-organism outlook."
- [[paper.nature-biotech.figures]] — figure asset catalog / workflow (final exports
  land here; the **Figure d/e** section lives in this note, above).
- [[torchcell.models.equivariant_cell_graph_transformer]] — $H$, $\mathcal T_\psi$,
  $\mathcal R_\phi$ definitions.
- [[scratch.2025.01.16.143744]], [[experiments.smf-dmf-tmf-001.results]],
  [[experiments.002-dmi-tmi]] — the lookup $\ge$ sequence-embedding evidence.
- External: Huh et al., *The Platonic Representation Hypothesis*, ICML 2024;
  Ahlmann-Eltze et al., *Nat Methods* 2025 (perturbation transformers vs linear).

## 2026.07.03 - Canonical bit estimates for Fig 1c (and how to tighten them)

This is the accounting actually rendered in **Fig 1c** and the Methods "Encoded entities"
paragraph (`sections/methods.tex`). It is the *universe-scale upper-bound* version and
supersedes the per-genome $10^6$-$10^7$ numbers above (those are *per-genome effective* bits).
Everything is order-of-magnitude; the ~100x gap dwarfs the slop.

**Entity universe (universe of things) ~ $10^{10}$ bits (upper bound).**

- One yeast/fungal genome: $L\approx1.2\times10^7$ bp; DNA at 2 bits/bp $\Rightarrow 2L\approx2.4\times10^7$ bits/genome.
- Upper bound on the yeast-relevant entity space = a large fungal-genome collection. Use the
  fungal genomes from Chao et al. 2025 (`chaoPredictingDynamicExpression2025`, the *Shorkie*
  fungal DNA language model).
  - **VERIFY THE COUNT.** The abstract states **165** fungal genomes were used for *pretraining*;
    the **1,342** figure (used in the paper text now) is presumably a larger candidate/collection
    set - confirm against the paper body and switch the manuscript number if needed.
  - Either way it is $\sim10^{10}$: $1342\times2.4\times10^7\approx3\times10^{10}$;
    $165\times2.4\times10^7\approx4\times10^9$.
- $\Rightarrow$ entity universe $\approx 10^{10}$ bits.

**Label space (space of instances) ~ $10^{8}$ bits (upper bound).**
Count measurements across the integrated datasets (rough, per the data tables):

- fitness: one scalar per strain (SGA single/double/triple; $\sim10^6$-$10^7$ values);
- expression: one scalar per gene per strain (knockout transcriptomes; $\sim10^3$ strains $\times$ 6,607 genes $\approx10^7$);
- protein abundance: one scalar per gene per condition ($\lesssim10^6$);
- morphology: a fixed trait set per strain ($\sim10^3$ strains $\times\sim10^2$-$10^3$ traits $\approx10^6$).

Total $\sim10^7$ measurements $\times\sim10$ bits/measurement $\Rightarrow \sim10^8$ bits.

**The gap.** $10^{10}$ entity bits vs $10^{8}$ label bits $=\sim100\times$ (~2 OOM). The labels
cannot pay for the entity representation, so cut the model at $H$ (encode on the abundant side,
spend labels only on operator + decoder). This is the one-line justification for ENC/PERT/DEC.

**How to make this rigorous (leave as SI / future work).**

1. *Entity bits - effective, not raw.* Replace $2L$/genome (which double-counts shared ancestry
   across the fungal genomes) with the entropy/MDL of the genome *collection*: per-column entropy
   over MSAs of the 6,607 proteins (functional content), or a kmer/compression estimate of the
   pangenome; and count *segregating* sites $\times$ mean per-site allelic entropy for in-panel
   variation. Expect this to lower the entity number but keep it $\gg$ labels.
2. *Label bits - effective, noise- and redundancy-aware.* Per assay, $b=\tfrac12\log_2(1+\mathrm{SNR})$
   from replicate CV (not a flat 10 bits); then subtract redundancy - interaction matrices are
   low-rank, so independent supervision $\approx r\cdot(\#\text{genes})\cdot b$ with $r$ from the
   SVD (tens-hundreds), not $N\times b$.
3. *Identifiability statement.* Frame as $B_{\text{eff}}\gtrsim C_{\text{fit}}$; end-to-end forces
   $C_{\text{fit}}\sim P_{\text{seq-encoder}}\sim10^8$-$10^{10}$ (violated by 2-4 OOM); the factored
   cut sets $C_{\text{fit}}=|\psi|+|\phi|\ll B$. Full version = the "core inequality" + the
   "Placeholders to pin" table above.

## 2026.07.12 - Measured accounting: persistent entities vs contingent observations (supersedes the estimates above)

**What changed.** The Fig 1c bit counts are no longer hand-derived. They are now *measured* from the
public archives by a committed script
([[experiments.016-information-accounting.scripts.persistent_entity_corpus_sizes]]) and rendered as
**Table S2** (`paper/nature-biotech/sections/tab-entity-corpora.tex`, auto-generated). The argument is
written up as **Supplementary Note 5: persistent entities and contingent observations**
(`sections/backmatter.tex`); the classical-ML note moved 5 → 6 to sit directly after it as its
empirical counterpart.

**Naming (final).** The two worlds are **persistent entities** ($\mathcal U$) and **contingent
observations** ($\mathcal I$). This replaces "universe of things / universe of instances" and
"encoded entities / labeled instances" in figure and paper text. The word doing the work is
*persistence*: a protein sequence is the same object in a glucose assay and an oxidative-stress
assay; the phenotype is not. (Rejected: "purity" — it implies sequence databases are error-free,
which they are not.)

### The measure, and the Kolmogorov-vs-Shannon question

$$L_C(D) = 8\,|C(s(D))| \quad\text{bits}$$

for corpus $D$, fixed serialization $s$, fixed lossless compressor $C$. **It is not an information
content.** It is *both*:

- a computable **upper bound on Kolmogorov complexity**: $K(D)\le L_C(D)+K(C^{-1})+O(1)$, because
  (compressed file + decompressor) is a program that outputs $D$; and
- an exact **Shannon codelength**: $C$ is injective with self-delimiting output, so by Kraft
  $q_C(\cdot):=2^{-L_C(\cdot)}$ is a sub-probability and $L_C(D)=-\log_2 q_C(s(D))$ — the
  self-information of $D$ under *gzip's* implicit model $q_C$, not under the true source. The excess
  over the source entropy is exactly $D_{\mathrm{KL}}(\text{source}\,\|\,q_C)$.

So the user's intuition ($-\log p$ ↔ compressibility) is right; the missing qualifier is *under which
model $p$*. Both worlds are measured with the **same** $(s,C)$, so the **ratio** is the claim, to
order of magnitude, and no single number is asserted as an information content. This is
Proposition 5 in the SI, with a two-step proof.

Terminology: say **compressed length / compressed size**, not "gzip codelength" (*codelength* reads
as source code outside information theory) and not "storage footprint" (throws away the
information-theoretic content entirely).

### Measured (2026.07.12)

| Modality | Corpus | GB | bits |
| --- | --- | --- | --- |
| Protein | UniProtKB/TrEMBL | 40.5 | 3.2 × 10¹¹ |
| Protein | UniProtKB/Swiss-Prot | 0.1 | 7.5 × 10⁸ |
| DNA / nucleotide | NCBI `nt` | 884.1 | **7.1 × 10¹²** |
| RNA | NCBI RefSeq RNA | 67.4 | 5.4 × 10¹¹ |
| Small molecule | PubChem Compound | 115.8 | 9.3 × 10¹¹ |
| Structure | wwPDB mmCIF | 89.2 | 7.1 × 10¹¹ |
| **Persistent entities, total** | | **1,197.1** | **9.6 × 10¹² ≈ 10¹³** |
| Contingent observations (34 studies) | | 1.4 | ~1.1 × 10¹⁰ ≈ 10¹⁰ |

**Ratio ≈ 10³ (three orders of magnitude).** Sources, routes and traps:
[[experiments.016-information-accounting.scripts.persistent_entity_corpus_sizes]].

**Corroboration.** `nt` holds 4.06 × 10¹² nucleotides → a four-letter alphabet needs ≥ 8.1 × 10¹²
bits; the archive ships in 7.1 × 10¹². The compressed size recovers 87% of a counting bound that
never mentions a compressor. Two unrelated routes agree, so the measure tracks sequence content, not
file formatting.

### The trap this created — and how the SI avoids it

The old accounting said labels ≈ $10^8$ bits and encoder capacity ≈ $10^8$–$10^{10}$ params, so
"labels cannot pay for the encoder" was a 2–4 OOM gap. The **new** number is $10^{10}$ *compressed
bits* on the label side — which, taken naively, looks like the labels have *caught up* with encoder
capacity and the factorization is no longer forced. **It has not**, and the SI is careful:

- $L_C(\mathcal I) \approx 10^{10}$ is **compressed storage**.
- $B_{\text{eff}} \approx 10^6$–$10^7$ is **usable supervision** (noise-discounted, low-rank-discounted;
  the `2026.06.23` B1/B2 analysis above).

These are *different quantities* and must never be swapped. The SI therefore makes **two separate
claims**: (1) a *measured* corpus-scale ratio, $10^{13}$ vs $10^{10}$, assumption-free — this is what
Fig 1c shows; and (2) the *identifiability* argument, which uses $B_{\text{eff}}$ and is an explicit
MDL heuristic. Methods and Note 5 both state the discount out loud.

### The second asymmetry (the one that actually matters)

Scale is the weaker half. **Persistence** is the argument: an entity representation is a function of
the entity alone, so one sequence informs the encoder for *every* instance in which that entity
appears; a phenotype record $y=f(\{x_i\}_{i\in p},p,\varepsilon,a)+\eta$ constrains $f$ at **one**
point of the context space and nowhere else. *Scale is an argument about sample size; persistence is
an argument about what a bit buys.*

### The objection to pre-empt

*"You only need the yeast entities, and the yeast proteome is ~3 × 10⁶ residues ≈ 10⁷ bits — smaller
than the phenotype corpus."* Correct arithmetic, inverted conclusion: an entity encoder is **not fit
on the yeast slice**, it is fit on the whole archive and *applied* to it, because a protein sequence
means the same thing in every organism (fungal DNA-LMs, e.g. `chaoPredictingDynamicExpression2025`,
are built exactly this way). That transfer **is** what persistence buys, and it is why the full
archive is the right comparand. This objection is answered explicitly in Note 5.

### Open

- **The one provenance gap.** The contingent side (~1.4 × 10⁹ compressed bytes) is still computed on
  gilahyper. Move that script into `experiments/016-information-accounting/scripts/` and regenerate
  **Table S5** (`tab:datasets`) from it *with a compressed-size column*. Table S5 currently lists 27
  studies (~4.4 × 10⁷ genotypes); the gilahyper table lists 34 (~7.9 × 10⁷ instances). Until they
  agree, Note 5 deliberately says "~10⁷–10⁸ instances" rather than quoting 34 against a 27-row table.
- **Fig 1c label.** The panel's top already reads ≳10¹³; its **bottom still reads ~10⁸ and must become
  ~10¹⁰**. The draw.io XML is embedded in
  `notes/assets/drawio/Fig1-torchcell-overview.drawio.svg` (open the `.svg` directly in draw.io).
- **Optional.** Panel c draws DNA / RNA / Protein / Small Molecules but no **Structures** box, while
  Table S2 has a wwPDB row (7.1 × 10¹¹ bits, ~7% of the total). Adding it would align figure and
  table; the ≳10¹³ headline holds either way.
