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
on a thin perturbation operator** — with a simple information accounting. The
figure that renders this argument is described in **Figure d/e** below; final
exports live in [[paper.nature-biotech.figures]].

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
|---|---|
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
|---|---|---|---|
| $V,\ \bar h$ | segregating sites + per-site entropy across our strain panel | VCF/alignment of the related *cerevisiae* genomes → count variable sites, compute allele-frequency entropy | the genome papers we hold `[PIN: list them]` |
| A2 bits | functional sequence content per genome | per-column entropy over MSAs of the 6,607 proteins (or a conservation proxy) | proteome + an MSA/conservation tool |
| $b$ | effective bits per phenotype measurement | $\log_2(\sigma_{\text{signal}}/\sigma_{\text{noise}})$ from replicate CV, per assay | Costanzo/Kuzmin replicate stats; expression/morphology noise |
| $r$ | effective rank of the interaction signal | SVD/PCA of the gene×gene (and trigenic) interaction matrix; count factors to ~90% variance | in-repo interaction data |
| $P_{\text{seq}}$ | capacity of a candidate end-to-end encoder | parameter count of the smallest credible sequence→phenotype model | ESM2 / DNA-LM sizes |

The argument's **shape** does not depend on these; they sharpen the headline ratios
$B/A$ and $B/C$ from "2–4 OOM" to specific numbers for the Methods/Discussion.

### Figure d/e: things on top, the contingent on the bottom

**Chosen direction (current draw.io).** The *universe of things* — DNA, RNA,
protein, small molecules — sits on top and fans down, composing per cell into a
**set of entity tokens** $\{\,\square\square\square\,\}$. A cell/instance is thus a
**set of entities drawn from $\mathcal U$**, $c=\{e_1,\dots,e_k\}$, and the
per-modality **pretrained encoders** $\mathrm{Embed}$ (inside $F_\theta$) map each
cell to its representation — $c_u\!\to\!h_u$, $c_v\!\to\!h_v$ (whole-cell vectors
$h_u:=h_{\mathrm{CLS}}^{(u)}$). This draws Corollary 3 (instances built from *sets of
things* via other pretrained models) with no platonic vocabulary.

**What it's missing: the contingent.** As drawn it stops at $h_u,h_v$ — pure
representation. The **labels never appear**, yet labels are the whole reason the
instances side exists; they come from the *contingent* (the experiments actually
measured). The panel shows $\mathcal U\!\to\!H$ but not $H\!\to\!\mathcal Y$, so the
cut the math is about is invisible.

**The fix — mirror it: the contingent on the bottom, rising to form the label
vector** $y$, symmetric to the things fanning down:

```
        UNIVERSE OF THINGS         dense · pretrained · ~10^7+ bits
      DNA     RNA     protein     small molecules
         \     \       /       /            fan DOWN
              {  ▭ ▭ ▭ ▭ ▭  }     set of entity tokens / cell
                     │  Embed / F_θ   (pretrained encoders)
           c_u ─▶ h_u        c_v ─▶ h_v          representation H
  ═══════ cut: H meets y   (operator T_ψ(·,S) + decoder R_φ) ═══════
           c_u ◀─ y_u        c_v ◀─ y_v          label vectors
                     │  measured phenotypes
           [ fitness · interaction · expression · morphology ]
         /     /       \       \            fan UP
        THE CONTINGENT             sparse · measured · ~10^6 bits
```

Each cell $c$ lives on **both** sides — encoded from things ($h$, top) and measured
in the contingent ($y$, bottom). The operator $\mathcal T_\psi$ (then readout
$\mathcal R_\phi$) predicts $\hat y$ from $H$ (+ perturbation $S$); the loss pins it
to the measured $y$. **That middle line is
the $\mathcal U\,\vert\,\mathcal I$ cut — the whole note in one picture.** Draw the
top band wide/dense and the bottom band thin/sparse so the bit-budget asymmetry
($\sim$10⁷ things-bits vs $\sim$10⁶ label-bits) reads without a caption.

**Verdict on the d/e split: correct and forced.** d = things$\,\to H$ (the encoder
$F_\theta$); e = $H\to\hat y$ via the operator $\mathcal T_\psi$ + readout $\mathcal
R_\phi$, supervised by the contingent —
the two panels are the two sides of the core inequality.

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
