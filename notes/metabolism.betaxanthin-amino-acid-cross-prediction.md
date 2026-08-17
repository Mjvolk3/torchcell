---
id: 8orahn4ln4ora3zhsk3tmxp
title: Betaxanthin Amino Acid Cross Prediction
desc: ''
updated: 1786929476832
created: 1786929476832
---

**Hypothesis.** Betaxanthin production and amino-acid metabolism are pathway-coupled, so
their genome-wide deletion datasets should help predict one another. Betaxanthin is made
from L-tyrosine (L-tyrosine → L-DOPA → betalamic acid → betaxanthin; pathway map in
[[metabolism.beta-carotene-and-betaxanthin]]), and the final condensation is a
spontaneous Schiff-base reaction between betalamic acid and an amino acid or amine -- the
amino-acid pool is literally a substrate. The two datasets over the same deletion
collection:

- [[torchcell.datasets.scerevisiae.cachera2023]] -- CRI-SPA betaxanthin level per
  deletion strain (4735 records, `MetabolitePhenotype`).
- [[torchcell.datasets.scerevisiae.mulleder2016]] -- amino-acid metabolome per deletion
  strain (WS9, `MetabolitePhenotype`).

A model trained on one should transfer signal to the other (shikimate flux, tyrosine
availability, amine pools are shared latent causes). The supervised-learning notation
below is the formal setup for that cross-prediction study; it is the same notation used
in the manuscript (`paper/nature-biotech/editing.pdf`, Methods).

## 2026.08.02 - Supervised Learning Notation (slide version)

Notation matches `paper/nature-biotech/sections/methods.tex` (Problem setup, Eq. 1--2, and
Table 1 "Notation"). Supersedes the tilde notation
($\tilde{G},\tilde{E},\tilde{P}$) in [[scratch.2025.07.07.183123-torchcell-basic-supervised-formulation]].
Rendered one slide per page at 16:9, into the usual PDF output directory:

```bash
cd notes
F=metabolism.betaxanthin-amino-acid-cross-prediction
pandoc -s "$F.md" -o "assets/pdf-output/$F.pdf" \
  --pdf-engine=tectonic -M title="" \
  -V documentclass=extarticle -V fontsize=20pt -V pagestyle=empty \
  -V geometry:'paperwidth=13.333in,paperheight=7.5in,margin=1in'
```

\newpage

### The learning problem

$$
\hat{f}_\theta:\ \mathcal{G}\times\mathcal{E}\times\mathcal{P}\ \longrightarrow\ \mathcal{Y}
$$

$$
\hat{\Theta}=\arg\min_{\Theta}\ \mathbb{E}_{(G,\varepsilon,p,y)\sim D}
\big[\mathcal{L}\big(\hat{f}_\theta(G,\varepsilon,p),\,y\big)\big]
$$

\newpage

### Where

- $G=(N,E)$: cell graph on $N=6607$ gene nodes, relation-$k$ adjacency $A^{(k)}\in\{0,1\}^{N\times N}$
- $\varepsilon$: environment (media, temperature)
- $p=\{(e_t,\tau_t,m_t)\}_{t=1}^{M}$: perturbation set -- entity, type, magnitude
- $\mathcal{G},\ \mathcal{E},\ \mathcal{P},\ \mathcal{Y}$: cell-graph, environment, perturbation, and phenotype spaces
- $y\in\mathcal{Y}$: observed phenotype; $\hat{y}$ predicted
- $D$: data law over $(G,\varepsilon,p,y)$
- $\mathcal{L}$: loss function
- $\Theta=(\theta,\psi,\phi)$: learnable parameters

\newpage

### The model factors: encode, perturb, decode

$$
\hat{f}_\theta(G,\varepsilon,p)=
\underbrace{\mathcal{R}_\phi}_{\text{DEC}}\big(
\underbrace{\mathcal{T}_\psi}_{\text{PERT}}(
\underbrace{F_\theta(G,\varepsilon)}_{\text{ENC}},\,p)\big)
$$

- $F_\theta$ (ENC): cell encoder, $H=F_\theta(G,\varepsilon)$
- $\mathcal{T}_\psi$ (PERT): perturbation operator, $H_{\mathrm{pert}}=\mathcal{T}_\psi(H,p)$
- $\mathcal{R}_\phi$ (DEC): decoder, $\hat{y}=\mathcal{R}_\phi(H_{\mathrm{pert}})$

The environment enters the **encoder**: $H$ represents this cell *in this condition*, so
the perturbation operator acts on a condition-aware encoding and can change it
accordingly. Abundant entity data trains $F_\theta$; the scarce phenotype labels need only
fit $\mathcal{T}_\psi$ and $\mathcal{R}_\phi$.

\newpage

## Layman version (same content, non-ML audience)

Alternate slides for a talk where the audience is biology-first. Same symbols, no ML
vocabulary: nothing here says "loss", "distribution", "parameters", or "objective".

\newpage

### What we are actually asking for

We want one function that takes a **cell**, a **growth condition**, and a **perturbation**
(a change to the genome), and tells us **what the cell does**. Training means turning the
model's internal dials until its predictions sit as close as possible to what we measured
in the lab.

$$
\underbrace{\hat{\Theta}}_{\substack{\text{the dial settings}\\\text{we keep}}}=
\arg\min_{\Theta}\ \underbrace{\mathbb{E}_{(G,\varepsilon,p,y)\sim D}}_{\text{averaged over every experiment}}
\Big[\ \underbrace{\mathcal{L}}_{\substack{\text{how far off}\\\text{we were}}}\big(
\underbrace{\hat{f}_\theta(G,\varepsilon,p)}_{\text{the prediction}},\
\underbrace{y}_{\text{the measurement}}\big)\Big]
$$

\newpage

### What goes in

- $G$ -- **the cell we start from.** Wild-type yeast: its 6607 genes, and the known links
  between them (regulation, physical interaction, metabolism).
- $\varepsilon$ -- **the growth condition.** Which medium, what temperature.
- $p$ -- **the perturbation**, the change we make to the genome. A list, each entry saying
  *which* gene ($e_t$), *what kind* of change ($\tau_t$: deleted, overexpressed, ...), and
  *how much* ($m_t$). Deleting two genes is a list of length two.

\newpage

### What comes out, and how we grade it

- $y$ -- **what we measured** in the lab for that strain (e.g.\ how fast it grew).
  $\hat{y}$ is what the model predicted instead.
- $D$ -- **the pile of real experiments** we learn from: every
  (cell, condition, perturbation, result) record we have.
- $\mathcal{L}$ -- **the scorecard.** One number saying how far the prediction was from
  the measurement. Training pushes it down.
- $\Theta$ -- **the dials.** All the internal numbers the model is free to adjust; there
  is one group per stage below.
- $\mathcal{G},\mathcal{E},\mathcal{P},\mathcal{Y}$ -- **"every possible one"** of the
  above: every cell, every condition, every change, every phenotype. The curly letters
  just mean we are talking about the whole range, not one example.

\newpage

### How the model is put together

$$
\hat{f}_\theta(G,\varepsilon,p)=
\underbrace{\mathcal{R}_\phi}_{\substack{\text{3. report}\\\text{the phenotype}}}\big(
\underbrace{\mathcal{T}_\psi}_{\substack{\text{2. apply the}\\\text{perturbation}}}(
\underbrace{F_\theta(G,\varepsilon)}_{\substack{\text{1. read the cell}\\\text{in its condition}}},\,p)\big)
$$

Reading the cell is the expensive step, and we do it **once** per condition. The
perturbation is then applied to that reading rather than to the cell, so millions of
strains reuse a single reading. Because the reading already knows the growth condition, the same
deletion is free to matter in one medium and not in another. That is the whole trick: the
plentiful data (genomes, networks) pays for step 1, and the scarce data (measured strains)
only has to pay for steps 2 and 3.

\newpage

### What changed from the old slide

| Old | New | Why |
|---|---|---|
| $\tilde{\mathcal{G}}$, $\tilde{G}$ | $\mathcal{G}$, $G=(N,E)$ | tildes dropped; $\tilde{A}^{(k)}$ now means row-normalized adjacency |
| $\tilde{\mathcal{E}}$, $\tilde{E}$ | $\mathcal{E}$, $\varepsilon$ | $E$ is the edge set of $G$; $\varepsilon$ frees it |
| $\tilde{\mathcal{P}}$, $\tilde{P}$ | $\mathcal{P}$, $p$ | $\mathbf{P}$ is the perturbation token matrix; $p$ is the perturbation set |
| $\tilde{P}$ = "perturbation operator" | $\mathcal{T}_\psi$ = perturbation operator | the operator is the learned map, not the data |
| $\hat{\theta}=\arg\min_\theta$ | $\hat{\Theta}=\arg\min_\Theta$, $\Theta=(\theta,\psi,\phi)$ | one symbol for all three factors' parameters |

Also changed here, relative to the current `methods.tex` draft (Eq. 2): the environment
$\varepsilon$ moves from the decoder into the encoder,
$\mathcal{R}_\phi(\mathcal{T}_\psi(F_\theta(G),p),\varepsilon)\ \Rightarrow\
\mathcal{R}_\phi(\mathcal{T}_\psi(F_\theta(G,\varepsilon),p))$, so that PERT operates on a
condition-aware encoding. **The manuscript has not been updated to match** -- see the
handoff note below.

\newpage

### Paper locations still carrying $\varepsilon$ in the decoder

- `sections/methods.tex:24` -- Eq. 2 (`eq:factor`), status `todo`
- `sections/methods.tex:76-80` -- Table 1 rows for $F_\theta$, $\mathcal{R}_\phi$, status `todo`
- `sections/methods.tex:267,283` -- operator/amortization paragraphs, status `todo`
- `sections/backmatter.tex:86,144,148` -- Supplementary Note on functional equivalence,
  status **`tent`** (author-approved; needs explicit go-ahead before editing)

Amortization argument under the change: encode-once becomes encode-once **per
(reference, condition) pair**. Conditions are few and strains are ${\sim}10^{4\text{--}6}$,
so "few references, many perturbations" survives, but the wording in Methods and in the
Supplementary Note should say so explicitly rather than implying one encoding per genome.

## 2026.08.16 - Graduated from scratch

Moved here from `notes/scratch.2026.08.02.235304-supervised-learning-notation.md` (scratch
original deleted per the scratch-note policy). The rendered slide PDF was renamed to match:
`notes/assets/pdf-output/metabolism.betaxanthin-amino-acid-cross-prediction.pdf`. The
pandoc command above regenerates it.
