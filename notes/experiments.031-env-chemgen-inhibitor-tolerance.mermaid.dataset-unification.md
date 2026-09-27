---
id: 8kp42tvmqxdfz6anbk9opw
title: Dataset Unification
desc: ''
updated: 1790500000000
created: 1790500000000
---

## 2026.09.27 - One row per dataset: what it measures, what it reports, how it joins

Read left to right, one row per served dataset. The left column is what the record IS, the
middle is what it REPORTS, and the right is the common form both are mapped into. The common
form is the same for every row, which is what "unified" means here: one input tuple and one
scalar target, with the per-source differences absorbed by a token rather than by a separate
model.

The figure carries the structure; the per-dataset numbers are in the companion table (t11 of
`notes-tex/031-unified-representation`, generated from `dataset_distributions.csv` and
`cross_dataset_pair_overlap.csv`). Keeping the counts out of the boxes is what lets the
diagram render at a readable size across the text block.

Every quantity is measured by a script under
`experiments/031-env-chemgen-inhibitor-tolerance/scripts/`. The three operations in the right
column are the only per-source transformations, and each exists because something was
measured, not because a dataset felt different:

- **orient** multiplies by the source's declared sick-sign. Without it two records measuring
  the same gene under the same compound carry opposite labels. Measured in
  `dataset_joinability.py`: the declared polarity predicts all 10 pairwise correlation signs,
  and Hillenmeyer HET against Hoepfner reads -0.118 over 170,837 shared cells as served.
- **standardize** divides by the source's own training-split standard deviation. The served
  standard deviations span 0.38 to 7.50, a factor of 20, so a pooled squared-error loss
  without this is a Wildenhain loss.
- **token** is the per-entry source one-hot. It carries the residual scale and offset and
  every physical field, because each physical field is constant within a dataset
  (`physical_axis_coverage.py`).

Mermaid escapes a literal `<` into `&lt;` before KaTeX sees it, which prints as `lt;`. The
comparisons below are written `\lt` and `\gt` for that reason.

Render with
`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.031-env-chemgen-inhibitor-tolerance.mermaid.dataset-unification.md`.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 12, "rankSpacing": 34}}}%%
flowchart LR
  classDef rec fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef out fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef uni fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef lost fill:#F8CECC,stroke:#A24A46,color:#1F1D1A

  VI["$$\begin{gathered}\textbf{Vanacloig 2022}\\\ \text{haploid, sensitized host}\\\ 3{,}598 \times 41,\ \text{IC30, anaerobic}\end{gathered}$$"]:::rec
  VO["$$\begin{gathered}\log_2(\text{inhib}/\text{ctrl})\\\ \text{sick} \lt 0,\ \text{sd } 0.70\end{gathered}$$"]:::out
  VI --> VO

  HEI["$$\begin{gathered}\textbf{Hillenmeyer HET}\\\ \text{het diploid, one of two}\\\ 5{,}810 \times 290,\ \text{fixed molar}\end{gathered}$$"]:::rec
  HEO["$$\begin{gathered}\log_2(\text{ctrl}/\text{treat})\\\ \text{sick} \gt 0,\ \text{sd } 0.38\end{gathered}$$"]:::out
  HEI --> HEO

  HPI["$$\begin{gathered}\textbf{Hoepfner 2014}\\\ \text{het AND hom diploid}\\\ 5{,}839 \times 148,\ \text{IC30 molar}\end{gathered}$$"]:::rec
  HPO["$$\begin{gathered}\text{MADL sensitivity}\\\ \text{sick} \lt 0,\ \text{sd } 1.29\end{gathered}$$"]:::out
  HPI --> HPO

  WI["$$\begin{gathered}\textbf{Wildenhain 2015}\\\ \text{haploid, narrow panel}\\\ 242 \times 5{,}170,\ 20\ \mu\text{M fixed}\end{gathered}$$"]:::rec
  WO["$$\begin{gathered}\text{OD}_{600}\ z\text{-score}\\\ \text{sick} \lt 0,\ \text{sd } 7.50\end{gathered}$$"]:::out
  WI --> WO

  UNI["$$\begin{gathered}\textbf{common form, every row}\\\ x = (d, C_e, M_e, b_e, \mu_e \ell_e, t)\\\ \tilde y = s_k (y - \mathrm{med}_k) / \mathrm{sd}_k\\\ s_k\ \text{declared sick-sign}\\\ \mathrm{sd}_k\ \text{from TRAIN only}\end{gathered}$$"]:::uni
  VO -- "$$s = -1$$" --> UNI
  HEO -- "$$s = +1$$" --> UNI
  HPO -- "$$s = -1$$" --> UNI
  WO -- "$$s = -1$$" --> UNI

  NOTE["$$\begin{gathered}\textbf{what this does not buy}\\\ \text{oriented agreement on shared}\\\ \text{cells is } 0.001\ \text{to}\ 0.198\\\ \text{a shared SCALE is justified,}\\\ \text{a shared RESPONSE is not}\end{gathered}$$"]:::lost
  UNI --> NOTE

  DROP["$$\begin{gathered}\textbf{HOM dropped}\\\ \text{1.09M records}\\\ 80\%\ \text{of its genes are in HET}\end{gathered}$$"]:::lost
  UNI -.-> DROP
```

**What the diagram asserts, and the evidence.**

- The four rows map onto ONE input tuple with no per-source input branch. The only per-source
  objects are a scalar sign, a scalar scale and a token.
- The sign is not a convenience. As served, Hillenmeyer HET and Hoepfner correlate at -0.118
  over 170,837 shared (gene, compound) cells, and Hillenmeyer HOM and Hoepfner at -0.198 over
  100,276. Pooling without orienting trains on contradictory labels at that scale.
- Standardizing uses the TRAIN split's statistics per source, never the full dataset, so no
  held-out value informs the scale.
- The last box is the honest limit. After orienting, the best cross-dataset agreement on
  shared cells is 0.198 and the median is 0.068, so these datasets agree about a gene's
  general fragility far more than about its response to a particular compound. That is why
  the expected gain is a gene prior and a wider compound space rather than more labels for
  the same task.

**A note on how this file was nearly lost.** `dendron-cli note write` on an EXISTING note
replaces it with a bare frontmatter stub. It is for new notes only; an existing note is edited
in place. The rendered PDF, SVG and PNG under `notes/assets/pdf-output/` survived, so the
figure was recoverable, but the source would not have been.
