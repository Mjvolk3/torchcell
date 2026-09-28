---
id: 8kp42tvmqxdfz6anbk9opw
title: Dataset Unification
desc: ''
updated: 1790500000000
created: 1790500000000
---

## 2026.09.27 - One row per dataset: what it measures, what it reports, how it joins

Four rows, one per kept dataset, read left to right through four labeled layers. The layers
carry the color, so the caption can refer to them by letter instead of by hue:

| layer | color | what it holds |
|---|---|---|
| **a** | purple | the STRAIN and the CONDITION, which is what the record is |
| **b** | yellow | what the SOURCE REPORTS, its own readout in its own units and sign |
| **c** | orange | the TRANSFORM applied to reach the shared target, and the shared form itself |
| **d** | red | what is given up, and what the shared form does not buy |

The layer letters are bold lowercase in plain HTML rather than KaTeX, because KaTeX renders
`\textbf` in its own math font and the repo standard asks for bold Arial, matching the
matplotlib panel letters.

The flow is a to b to c: a record is a strain in a condition, the source reports it on its own
scale, and two per-source scalars map that onto a common target. d hangs off c and is
commentary, not a step.

**The orientation factor `o_k` is a sign flip on a continuous number.** It is NOT a threshold
and NOT a class label. Three sources already store a sick strain as negative and take
`o_k = +1`; the two Hillenmeyer arms store it as positive and take `o_k = -1`. After the flip,
a more negative value means a sicker strain in every dataset. Nothing is binarized anywhere in
this diagram or in the pipeline it describes; the task is regression on a real number.

**MADL** is the median absolute deviation logarithmic score, Hoepfner's per-experiment
readout: a strain's log abundance ratio minus the median over all strains in the sample,
divided by that set's median absolute deviation.

**HOM is the homozygous arm of Hillenmeyer 2008**, its homozygous-diploid deletion collection,
released as its own file with its own readout. It is dropped; its heterozygous sibling HET is
kept and already covers 80 percent of its genes.

Per-dataset numbers are in the companion table (t11 of
`notes-tex/031-unified-representation`). Keeping counts out of the boxes is what lets the
diagram render at a readable size across the text block.

Mermaid escapes a literal `<` into `&lt;` before KaTeX sees it, which prints as `lt;`. The
comparisons below are written `\lt` and `\gt` for that reason.

Render with
`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.031-env-chemgen-inhibitor-tolerance.mermaid.dataset-unification.md`.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 12, "rankSpacing": 32}}}%%
flowchart LR
  classDef rec fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef out fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef uni fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef lost fill:#F8CECC,stroke:#A24A46,color:#1F1D1A
  classDef hdr fill:#FFFFFF,stroke:#FFFFFF,color:#1F1D1A

  HA["<b>a</b>&nbsp;&nbsp;strain and condition"]:::hdr
  HB["<b>b</b>&nbsp;&nbsp;what the source reports"]:::hdr
  HC["<b>c</b>&nbsp;&nbsp;transform and shared form"]:::hdr
  HD["<b>d</b>&nbsp;&nbsp;what is given up"]:::hdr
  HA ~~~ HB ~~~ HC ~~~ HD

  VI["$$\begin{gathered}\textbf{Vanacloig 2022}\\\ \text{haploid, sensitized host}\\\ \text{4 deletions: query + PDR1, PDR3, SNQ2}\\\ 3{,}598\ \text{genes} \times 41\ \text{compounds, IC30}\end{gathered}$$"]:::rec
  VO["$$\begin{gathered}\log_2(\text{inhibitor}/\text{control})\\\ \text{sicker is more negative}\\\ \text{sd } 0.70\end{gathered}$$"]:::out
  VI --> VO

  HEI["$$\begin{gathered}\textbf{Hillenmeyer HET}\\\ \text{het diploid, one of two copies}\\\ 5{,}810\ \text{genes} \times 290\ \text{compounds}\end{gathered}$$"]:::rec
  HEO["$$\begin{gathered}\log_2(\text{control}/\text{treated})\\\ \text{sicker is more POSITIVE}\\\ \text{sd } 0.38\end{gathered}$$"]:::out
  HEI --> HEO

  HPI["$$\begin{gathered}\textbf{Hoepfner 2014}\\\ \text{het AND hom diploid, both arms}\\\ 5{,}839\ \text{genes} \times 148\ \text{compounds}\end{gathered}$$"]:::rec
  HPO["$$\begin{gathered}\text{MADL sensitivity score}\\\ \text{sicker is more negative}\\\ \text{sd } 1.29\end{gathered}$$"]:::out
  HPI --> HPO

  WI["$$\begin{gathered}\textbf{Wildenhain 2015}\\\ \text{haploid deletion, narrow gene panel}\\\ 242\ \text{genes} \times 5{,}170\ \text{compounds}\end{gathered}$$"]:::rec
  WO["$$\begin{gathered}\text{OD}_{600}\ z\text{-score}\\\ \text{sicker is more negative}\\\ \text{sd } 7.50\end{gathered}$$"]:::out
  WI --> WO

  UNI["$$\begin{gathered}\textbf{one input, one continuous target}\\\ x = (d, C_e, M_e, b_e, \mu_e \ell_e, t)\\\ \tilde y = o_k (y - \mathrm{med}_k) / \mathrm{sd}_k\\\ o_k = \pm 1\ \text{sign flip, NOT a threshold}\\\ \mathrm{med}_k, \mathrm{sd}_k\ \text{from TRAIN split only}\\\ \text{regression, nothing is binarized}\end{gathered}$$"]:::uni
  VO -- "$$o_k = +1$$" --> UNI
  HEO -- "$$o_k = -1\ \text{(flip)}$$" --> UNI
  HPO -- "$$o_k = +1$$" --> UNI
  WO -- "$$o_k = +1$$" --> UNI

  NOTE["$$\begin{gathered}\textbf{what this does not buy}\\\ \text{after orienting, agreement on shared}\\\ \text{cells is } 0.001\ \text{to}\ 0.198\\\ \text{a shared SCALE is justified,}\\\ \text{a shared RESPONSE is not}\end{gathered}$$"]:::lost
  UNI --> NOTE

  DROP["$$\begin{gathered}\textbf{Hillenmeyer HOM dropped}\\\ \text{the homozygous arm of the same paper}\\\ \text{1.09M records; HET already covers}\\\ 80\%\ \text{of its genes}\end{gathered}$$"]:::lost
  UNI -.-> DROP
```

**What the diagram asserts, and the evidence.**

- The four rows map onto ONE input tuple and ONE continuous target with no per-source input
  branch. The only per-source objects are a sign, a scale and a token.
- The sign is not a convenience. As served, Hillenmeyer HET and Hoepfner correlate at -0.118
  over 170,837 shared (gene, compound) cells, and Hillenmeyer HOM and Hoepfner at -0.198 over
  100,276. Pooling without orienting trains on contradictory labels at that scale.
- Standardizing uses the TRAIN split's statistics per source, never the full dataset, so no
  held-out value informs the scale.
- Layer D is the honest limit. After orienting, the best cross-dataset agreement on shared
  cells is 0.198 and the median is 0.068, so these datasets agree about a gene's general
  fragility far more than about its response to a particular compound. That is why the
  expected gain is a gene prior and a wider compound space rather than more labels.

**A note on how this file was nearly lost.** `dendron-cli note write` on an EXISTING note
replaces it with a bare frontmatter stub. It is for new notes only; an existing note is edited
in place.
