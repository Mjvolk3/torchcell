---
id: cblp5aoncbn1av54txaeaoq
title: Methods
desc: ''
updated: 1782935074133
created: 1661539594543
---

## 2026.06.28 - Methods draft (Fig 1 panels a–g)

Draft **Methods** for the Cell Graph Transformer (CGT) paper, organized to mirror the
conceptual figure (Fig 1, panels **a–g**). It opens with one deliberate **Notation**
reconciling the conventions of (i) machine learning on graphs, (ii) transformers /
attention, and (iii) constraint-based metabolic modeling, then gives one methods
subsection per panel. A detailed Supplementary section per panel will follow.

Standardizes and **supersedes** the tilde/bar notation in
[[scratch.2025.07.07.183123-torchcell-basic-supervised-formulation]]; consistent with
[[paper.information-accounting]], [[scratch.perturbation-operator-general]], and the model
note [[torchcell.models.equivariant_cell_graph_transformer]].

---

## Notation

**Convention — one decoration, one meaning.** Plain italics denote elements;
*calligraphic* denotes spaces and learned operators; **bold** denotes whole
matrices / tensors where it aids reading. The **tilde is reserved for the
row-normalized adjacency** $\tilde A$ (standard in graph ML) and is used for nothing
else; the $\tilde{(\cdot)}$ / $\bar{(\cdot)}$ decorations of earlier drafts are
dropped. Symbols are chosen to stay compatible with all three literatures.

**Sets, indices, spaces.**
$\mathcal U$ — universe of things (entities: DNA, RNA, protein, small molecules);
$\mathcal I$ — universe of labeled instances, $\mathcal I=\operatorname{supp}D$;
$\mathcal Y$ — phenotype space, $y\in\mathcal Y$ observed, $\hat y$ predicted.
Genes $V=\{g_1,\dots,g_N\}$, $\lvert V\rvert=N$, indices $i,j\in[N]$; relation
types $k\in[K]$; transformer layers $\ell\in[L]$, heads $a\in[h]$.

**Graphs (graph-ML convention).**
Cell graph $G=(V,E)$; for $K$ relation types, edge sets $E^{(k)}$ and adjacencies
$A^{(k)}\in\{0,1\}^{N\times N}$; degree $D^{(k)}=\operatorname{diag}(A^{(k)}\mathbf 1)$;
**row-normalized adjacency** $\tilde A^{(k)}=(D^{(k)})^{-1}A^{(k)}$ (each row a
distribution over neighbors — *the only tilde*); neighborhood
$\mathcal N_k(i)=\{j:A^{(k)}_{ij}=1\}$.

**Transformer / attention.**
Token states $H=(h_{\mathrm{CLS}},h_1,\dots,h_N)\in\mathbb R^{(N+1)\times d}$, width
$d$; layer states $H^{(\ell)}$; projections $Q=HW_Q,\ K=(\cdot)W_K,\ V=(\cdot)W_V$,
per-head width $d_k=d/h$; attention map
$\alpha^{(\ell,a)}=\operatorname{softmax}\!\big(Q^{(\ell,a)}K^{(\ell,a)\top}/\sqrt{d_k}\big)\in[0,1]^{N\times N}$
(rows sum to 1).

**Constraint-based metabolism (FBA convention).**
Stoichiometric matrix $S\in\mathbb R^{m\times r}$ ($m$ metabolites, $r$ reactions);
flux $v\in\mathbb R^{r}$; steady state $Sv=0$, bounds $v_{\min}\le v\le v_{\max}$;
gene→reaction association (GPR) $\rho:V\to 2^{[r]}$. *We reserve $S$ for the
stoichiometric matrix; perturbations are $p$ (below).* In torchcell metabolism enters
as the **bipartite** gene–reaction–metabolite graph induced by $(S,\rho)$ — a graph
annotation (§g), not (yet) a flux program.

**Encoders, model, perturbation.**
Entity encoder $\operatorname{Embed}:\mathcal U\to\mathbb R^{d}$ — a *pretrained*
DNA/protein/RNA/chem model, or a *learnable lookup* $\mathbf E\in\mathbb R^{N\times d}$.
Cell encoder $F_\theta$ ($=\operatorname{ENC}$): $H=F_\theta(G)$.
**Perturbation** $p$ — a finite **set** of perturbation tokens
$p=\{(e_t,\tau_t,m_t)\}_{t=1}^{M}$ with entity $e_t\in\mathcal U$, type $\tau_t$,
magnitude $m_t$; for genetic perturbations $e_t\in V$ and $\tau\in\{\mathrm{del},
\mathrm{OE},\mathrm{KD}\}$. Environment $\varepsilon\in\mathbb R^{d_\varepsilon}$
(media, temperature, …). Perturbation operator $\mathcal T_\psi$ ($=\operatorname{PERT}$):
$H_{\mathrm{pert}}=\mathcal T_\psi(H,p)$. Readout/decoder $\mathcal R_\phi$
($=\operatorname{DEC}$): $\hat y=\mathcal R_\phi(H_{\mathrm{pert}},\varepsilon)$.
Parameters $\Theta=(\theta,\psi,\phi)$; loss $\mathcal L$; data distribution $D$ over
$(G,\varepsilon,p,y)$.

**The supervised problem** (the tilde-free replacement for the earlier
$\hat f_\theta(\tilde G,\tilde E,\tilde P)$ formulation):
$$
\hat\Theta=\arg\min_{\Theta}\ \mathbb E_{(G,\varepsilon,p,y)\sim D}
\Big[\,\mathcal L\big(\underbrace{\mathcal R_\phi(\mathcal T_\psi(F_\theta(G),\,p),\,\varepsilon)}_{\hat y},\ y\big)\Big].
$$

> `[DECIDE]` rename **cell embedding → cell encoder** ($F_\theta$) for symmetry with
> $\operatorname{ENC}/\operatorname{PERT}/\operatorname{DEC}$. Flagged in panel d.

---

## a. An ontology-grounded schema and the knowledge graph

We type every experimental record with a schema $\Sigma$ **grounded in the Biolink
Model**, the standardized biomedical ontology: each schema class maps to a Biolink
category and each relation to a Biolink predicate (association). Records are validated
*programmatically* against $\Sigma$ at ingestion (typed classes + ontology-consistency
checks), so conformance is enforced, not assumed.

$\Sigma$ is a **directed acyclic graph** over types — an hourglass whose narrow waist
forces heterogeneous sources into a single typed structure. A raw record $x$ is
validated and emitted as a typed subgraph $\iota(x)\subset \mathcal K$ into a graph
database (knowledge graph $\mathcal K$, Neo4j). Datasets then become queryable over a
*shared* structure:

- **data-specific query** — collect an instance with its nearest typed edges
  (the orange→red link, both sides of panel a): returns one experiment with the edges
  needed to reconstruct its dataset context;
- **cross-experiment query** — collect all instances sharing a structural entity (e.g.
  a common **environment** node): joins experiments from *different* studies through
  shared schema entities.

Formally a query is a pattern $q$ over $\mathcal K$ and a dataset is its match set
$\{x:\iota(x)\models q\}$. This converts modular, heterogeneous literature / public
data into homogenized, queryable datasets — directly addressing the data-fragmentation
limitation that has held back deep learning for synthetic biology and metabolic
engineering.

> `[DECIDE]` **Is $\Sigma$ itself an ontology?** It is a TBox of abstract classes that
> denote the *types* (the things themselves), not instances, with programmatic axioms
> (validation), which is the case for calling it an ontology. For now we make the
> conservative claim — we **ground** $\Sigma$ in the standardized Biolink ontology and
> enforce it programmatically — and defer the stronger claim. (The figure labels this
> waist "Ontology"; the text clarifies it is a Biolink-grounded schema.)

---

## b. Modular datasets: a reference acted on by a perturbation

Each experiment is a tuple $(G,\varepsilon,p,y)$: a **reference** cell acted on by a
**perturbation**, observed under an **environment**, yielding a **phenotype**.

**Reference data (left).** The minimal reference is a **genome**: a sequence (FASTA)
plus an annotation (GFF) giving gene positions and sequence features — this alone is a
valid record. *Optional* annotations add structure over that genome: gene / regulatory
/ PPI networks $\{A^{(k)}\}_{k=1}^{K}$, metabolism (the bipartite graph from $(S,\rho)$),
and Gene Ontology. Annotations are optional precisely because they presuppose the
genome; absent it, there is nothing to annotate.

**Reference $+$ perturbation is the natural compression.** Data factorize as *few
references, many perturbations*: the yeast knockout library is one reference strain and
$\sim\!10^{4\text-6}$ deletion strains. Storing a whole genome per strain is redundant;
we store a shared reference $G$ and a perturbation $p$ over it. Natural isolates or
very different genomes are simply **new references** (store the whole genome). A state
carries an environment $\varepsilon$ when available; experiments always specify one.

**Experiment data (right).** An experiment requires a perturbation to the reference —
to the **genome** ($p$, the genotype) and/or to the **environment** ($\Delta\varepsilon$,
e.g. drug / inhibitor / tolerance studies) — and a phenotype $y$. The phenotype may be
**scalar** (fitness — global), **vector** (expression, protein abundance — gene-level /
local; morphology profiles), or **tensor**; the formulation is agnostic to the tensor
rank of $y$. Metabolism readouts give **metabolite concentrations**.

**Joining and aggregation** (SI figure; referenced, not in the main figure). A
mechanistic, statistical, explainable procedure merges instances — e.g. joining across
**environments**, or **recasting** a lethality call into a fitness measurement. Joining
both enlarges datasets and, by mapping heterogeneous readouts onto a unified phenotype
structure, **shrinks the decoder's output domain** $\mathcal Y$ (fewer distinct symbol
types $\mathcal R_\phi$ must emit), making the phenotype encoding more efficient — less
one-head-per-assay.

> `[REF]` link the SI data-joining/aggregation figure here once finalized.

---

## c. The universe of things and the labeled instances

Entity representations come from **pretrained models over the universe of things**:
DNA, RNA, and protein language models and small-molecule encoders (sequence or graph),
$\operatorname{Embed}:\mathcal U\to\mathbb R^{d}$. Small molecules need not be intrinsic
to the organism — these encoders represent *any* candidate molecule and place similar
molecules nearby in $\mathbb R^{d}$, so molecular geometry (intrinsic-dimension
similarity) is a usable prior: similar molecules are hypothesized to have similar
effects, a bridge for cross-entity (molecule↔gene) interaction.

We encode the reference cell with these embeddings into the cell representation $H$.
(Panel c shows the *minimal* idea — genome as sequence $+$ annotation; how graph
structure is encoded is deferred to §d/§f, to avoid clutter.)

**The cut.** Panel c marks the natural partition of the field: the **universe of
things** $\mathcal U$ (abundant; $\sim\!10^{10}$ bits) above, the **labeled instances**
$\mathcal I$ (scarce supervision; $\sim\!10^{8}$ bits) below — supervised data is always
the smaller side. The model is cut here so that data-rich representation learning lives
in $\mathcal U$ and the scarce labels fit only the thin operator; the quantitative
argument is in [[paper.information-accounting]].

---

## d. The Cell Graph Transformer (encode → operate → decode)

The CGT is the pipeline **multimodal cell input → cell encoder → perturbation operator
→ label decoder → multitask output + loss**:
$$
H=F_\theta(G),\qquad
H_{\mathrm{pert}}=\mathcal T_\psi(H,p),\qquad
\hat y_t=\mathcal R_\phi^{\,t}(H_{\mathrm{pert}},\varepsilon),
$$
with a sparse multitask objective over tasks $t$ (fitness, interaction, expression,
morphology) and per-task losses $\ell_t$,
$$
\mathcal L_{\mathrm{pheno}}=\sum_{t}w_t\!\!\sum_{b:\,y_t^{(b)}\ \text{observed}}\!\!\ell_t\big(\hat y_t^{(b)},y_t^{(b)}\big).
$$
The encoder $F_\theta$ is a graph-regularized transformer (§f); the operator
$\mathcal T_\psi$ and decoder $\mathcal R_\phi$ are detailed in §e and as task readouts.
One $H_{\mathrm{pert}}$ serves every head — *one representation, many phenotypes.*

---

## e. The perturbation operator (cross-attention, general case)

$\mathcal T_\psi$ applies the perturbation as **cross-attention from the whole cell to
the perturbation context**: every cell token is a query, and the keys/values are the
(small) perturbation set $p$. Embedding each token as
$c_t=r(e_t)+\mathbf t_{\tau_t}+m_t\mathbf w$ (with $r(e_t)=h_{j}$ for an in-cell gene,
else $\operatorname{Embed}(e_t)$), and $C=(c_1,\dots,c_M)$,
$$
\alpha_{i,t}=\operatorname*{softmax}_{t\in[M]}\!\Big(\tfrac{(h_iW_Q)\cdot(c_tW_K)}{\sqrt{d_k}}\Big),
\qquad
\Delta h_i=\sum_{t=1}^{M}\alpha_{i,t}\,(c_tW_V),
$$
$$
h_i^{\mathrm{pert}}=\operatorname{LN}\!\big(\tilde h_i+\operatorname{FFN}(\tilde h_i)\big),
\quad \tilde h_i=\operatorname{LN}(h_i+\Delta h_i),
\qquad H_{\mathrm{pert}}=\mathcal T_\psi(H,p).
$$
The update is a **soft weighted blend** over $p$ ($\sum_t\alpha_{i,t}=1$), not a
dictionary lookup. Because $p$ is a *set*, $\mathcal T_\psi$ is permutation-invariant in
the perturbation tokens and accepts any cardinality, and because each $\Delta h_i\in
\mathbb R^{d}$, the perturbation is a **transformation within cell-embedding space** —
**gene-equivariant** (permuting genes permutes $H_{\mathrm{pert}}$ identically). This is
what **generalizes the operator from gene deletions to any perturbation** — drugs,
inhibitors, environment shifts — by widening the token set $p$ with no architectural
change. Full derivation, properties, and the genotype/environment cartoon:
[[scratch.perturbation-operator-general]].

---

## f. Attention–graph regularization

A biological graph has a **square** adjacency $A^{(k)}\in\{0,1\}^{N\times N}$, and the
gene–gene attention map $\alpha^{(\ell,a)}\in[0,1]^{N\times N}$ has the **same shape** —
so we can align them directly. Row-normalize the graph to a per-node distribution and
penalize its divergence from the matched attention head:
$$
\mathcal L_{\mathrm{graph}}^{(k)}=\sum_{i\in\mathcal I_k}\operatorname{KL}\!\big(\tilde A^{(k)}_{i,:}\,\big\|\,\alpha^{(\ell_k,a_k)}_{i,:}\big),
\qquad
\mathcal L=\mathcal L_{\mathrm{pheno}}+\sum_{k=1}^{K}\lambda_k\,\mathcal L_{\mathrm{graph}}^{(k)},
$$
with one graph $k$ supervising one head $(\ell_k,a_k)$. Each gene network thus becomes a
**soft prior** baked into attention, rather than a hard architecture.

**Why regularization, not graph convolution over $A$.** (i) Edges carry experimental
evidence and are trustworthy where $A_{ij}=1$, but $A_{ij}=0$ often means *not yet
tested*, not *no interaction* — absence is uncertain. A hard graph operator commits to
those zeros; the soft KL prior only **biases** attention toward known edges and lets
data overrule it, so with enough data true interactions can be recovered even where
$A_{ij}=0$. (ii) As a *loss term* the same graphs can later be promoted from priors to
**labels** (edge prediction). (iii) **Interpretability**: a head with persistently large
$\alpha_{ij}$ on a non-edge ($A_{ij}=0$) flags a **candidate missing edge** the model
"keeps trying to wake up" — a hypothesis for an undiscovered interaction (to use
cautiously).

**Scope.** Only **gene–gene** graphs are regularized: in experiment 010 these are the
$K=9$ networks (physical, regulatory, TFLink, and six STRING channels), each mapped to
one head. Metabolism is **excluded** here — it is **bipartite** (gene×reaction, not
$N\times N$), so it has no square adjacency to KL against; it enters as a representation
annotation (§g), not an attention prior.

---

## g. The cellular experiment as a constrained world model

The innovation torchcell targets is a **representation of the whole cellular experiment,
from genome to environment** — nested containment
$$
\underbrace{\text{Genome}}_{\text{sequence}}
\subset \underbrace{\text{Gene Networks}}_{\text{multi-graph }\{A^{(k)}\}}
\subset \underbrace{\text{Metabolism}}_{\text{bipartite }(S,\rho)}
\subset \underbrace{\text{Environment}}_{\text{multi-set}} ,
$$
i.e. an attempt to reconstruct "everything in the well." Each scale contributes its
natural object — a sequence, a multi-graph, a bipartite (constraint-based) graph, and a
multi-set — unified in $H$.

We frame this as a **world model for the cell**: a deliberately limited world in which
we track and control as much of the system as possible, modeling the *system being
engineered* (the cell) and not only the dependent variables.

> `[DISCUSSION]` The following belong in the Discussion, flagged here for continuity.
> Prior design–build–test–learn (DBTL) pipelines in metabolic engineering largely model
> only the **dependent variables** (titer/yield) and treat the host cell as an
> uncontrolled background. But modifications and the rest of the cell interact
> dynamically and those effects **propagate** — most often negatively, sometimes
> positively — so ignoring the cell discards signal. Bringing the cell into the model is
> costly: the gain-per-bit is low and one must "go all the way" (a single dataset is
> unlikely to suffice unless very large); we therefore aim first to capture **general
> cross-cell patterns** before claiming engineering utility, which we hope to
> demonstrate by the paper's end. Tie to autonomous experimentation (UIUC iBioFoundry):
> a world model for cells under perturbation closes the DBTL loop
> (Learn→Design→Build→Test). Distinguishing experimental procedures / batch biases is
> currently absorbed into label readouts; with enough data these could be identified
> (future work).

---

## Open notation / scope decisions (collected)

- `[DECIDE]` perturbation symbol **$p$** (set) vs the model note's $S$ (gene set) — adopt
  $p$ project-wide so $S$ is free for the stoichiometric matrix; propagate to
  [[paper.information-accounting]] and the model note.
- `[DECIDE]` **cell embedding → cell encoder** ($F_\theta$) naming (panel d).
- `[DECIDE]` schema-as-ontology claim (§a) — conservative grounding for now.
- `[REF]` SI figures: data joining/aggregation (§b); per-panel SI sections (a–g).
