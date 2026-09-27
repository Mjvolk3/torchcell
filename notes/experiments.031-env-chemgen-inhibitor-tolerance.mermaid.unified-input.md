---
id: 7hn31rskxrceyz5dmal8npj
title: Unified Input
desc: ''
updated: 1790485855374
created: 1790485855374
---

## 2026.09.27 - One input representation and one decoder for four chemogenomic datasets

Every quantity in this diagram is measured, from the scripts under
`experiments/031-env-chemgen-inhibitor-tolerance/scripts/`. The boxes carry the tensor each
channel is, in the notation of the cell graph transformer notes
([[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]]): $N$ genes,
$d_h$ the hidden width, $S$ the perturbed set, $H_{\mathrm{pert}}$ the Type I output, $R_\phi$
a Type II readout. Render with
`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.031-env-chemgen-inhibitor-tolerance.mermaid.unified-input.md`.

Colors follow the draw.io palette: gray = the served records, purple = a per-record input
channel, yellow = a fixed table, orange = computation, blue = an optional branch, red = what
the representation cannot carry or deliberately drops.

The two decisions taken on 2026.09.27 are drawn in: the Hillenmeyer homozygous arm is dropped,
and the functional dose scales the existing cross-attention perturbation operator through
$\gamma(m_j)$ rather than entering as a separate feature. The physical fields have no channel
because, measured across the five datasets, every one of them is constant within a dataset
(`physical_axis_coverage.py`), so the dataset token already carries them.

Layout note: the top-level direction is `TB` and the four datasets are lines inside ONE box
rather than four boxes. Five parallel source boxes put five labels on one rank and rendered at
a 4.5-to-1 aspect ratio, which at text width prints the math near 1.5 pt.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial", "fontSize": "13px"}, "flowchart": {"nodeSpacing": 18, "rankSpacing": 34}}}%%
flowchart TB
  classDef src fill:#F5F5F5,stroke:#666666,color:#1F1D1A
  classDef rec fill:#E1D5E7,stroke:#846592,color:#1F1D1A
  classDef par fill:#FFF2CC,stroke:#BCA04C,color:#1F1D1A
  classDef comp fill:#FFE6CC,stroke:#BD8800,color:#1F1D1A
  classDef opt fill:#DAE8FC,stroke:#5B7AA6,color:#1F1D1A
  classDef lost fill:#F8CECC,stroke:#A24A46,color:#1F1D1A

  SRC["$$\begin{gathered}\textbf{four served datasets, 6.39M records}\\\ \text{Vanacloig 2022: haploid, 41 cpd, IC30}\\\ \text{Hillenmeyer HET: het diploid, 290 cpd}\\\ \text{Hoepfner 2014: both arms, 148 cpd}\\\ \text{Wildenhain 2015: haploid, 5{,}170 cpd}\end{gathered}$$"]:::src
  REC["$$\text{one served record}\quad (g, e) \mapsto y$$"]:::comp
  SRC --> REC

  GT["$$\begin{gathered}\textbf{genotype}\\\ d \in \{0, \tfrac{1}{2}, 1\}^{N},\ N = 6{,}607\\\ d_i = c_i / r_i\ \text{(copies / reference)}\\\ S = \{ i : d_i \lt 1 \},\ m_i = 1 - d_i\\\ \text{ploidy is the background level}\end{gathered}$$"]:::rec
  EN["$$\begin{gathered}\textbf{environment}\quad e = (C_e, M_e, b_e, \ell_e)\\\ C_e\ \text{dosed compounds, 5{,}472 keys}\\\ M_e\ \text{medium components, 37}\\\ b_e \in \{\mathrm{IC30}, \mathrm{fixed}, \mathrm{none}\}\\\ \ell_e = \log_{10} M,\ \text{mask}\ \mu_e \in \{0,1\}\end{gathered}$$"]:::rec
  TOK["$$\begin{gathered}\textbf{dataset token}\\\ t \in \{0,1\}^{K},\ K = 4\\\ \text{measurement scale and every}\\\ \text{within-dataset constant}\end{gathered}$$"]:::rec
  REC --> GT
  REC --> EN
  REC --> TOK

  PH["$$\begin{gathered}\text{physical fields } (T, \text{aer.}, \text{dur.}, \mathrm{pH})\\\ \text{constant inside every dataset:}\\\ \text{no channel, absorbed by } t\end{gathered}$$"]:::lost
  REC -.-> PH

  CG["$$\begin{gathered}\textbf{cell graph encoder}\\\ H^{(0)} \in \mathbb{R}^{(N+1) \times d_h}\\\ H^{(L)} = T^{(L)} \cdots T^{(1)}(H^{(0)})\\\ \mathrm{KL}(A_g \Vert \alpha^{(\ell,k)}),\ \text{9 graphs}\end{gathered}$$"]:::comp
  ME["$$\begin{gathered}\textbf{molecule encoder } f\\\ z = f(\mathrm{SMILES}) \in \mathbb{R}^{D_f}\\\ 12\ \text{registered},\ D_f \in [167, 2048]\\\ \text{FCFP4 count best measured}\\\ \text{one table: cpd, medium, Yeast9}\end{gathered}$$"]:::par
  EN --> ME
  CG --> OP

  OP["$$\begin{gathered}\textbf{perturbation operator (Type I)}\\\ H_{\mathrm{pert},i} = h_i + \sum_{j \in S} \gamma(m_j)\, \alpha_{ij} W_V h_j\\\ \gamma(1) = 1:\ \text{deletion data unchanged}\\\ \gamma(\tfrac{1}{2})\ \text{from the 648{,}977 pairs}\end{gathered}$$"]:::comp
  ZE["$$\begin{gathered}\textbf{environment context}\\\ z_E = [\, \sum_{c \in C_e} z_c \Vert \sum_{m \in M_e} z_m \Vert b_e \Vert \mu_e \ell_e \,]\end{gathered}$$"]:::comp
  GT --> OP
  ME --> ZE
  EN --> ZE
  ZE -. "$$\text{arm 2: extra K/V rows}$$" .-> OP

  MET["$$\begin{gathered}\text{metabolism branch, optional}\\\ \text{Yeast9: } 894\ \text{of}\ 1{,}378\ \text{through } f\\\ \text{the flux layer must read } d\end{gathered}$$"]:::opt
  MET -.-> ME

  DEC["$$\begin{gathered}\textbf{unified decoder (Type II)}\\\ z_S = \mathrm{pool}_{j \in S} H_{\mathrm{pert},j}\\\ \hat y = R_\phi([\, h_{\mathrm{CLS}} \Vert z_S \Vert z_E \Vert W_t t \,])\\\ \text{one scalar head; } y \text{ is } \log_2\text{ratio, } z, \text{ or sens.}\\\ \mathcal{L} = \sum_k w_k \sum_{b \in k} \ell(\hat y_b, y_b)\end{gathered}$$"]:::comp
  OP --> DEC
  ZE --> DEC
  TOK --> DEC
  MET -.-> DEC

  LOST["$$\begin{gathered}\text{not carried: cassette identity, barcode, collection}\\\ \text{dose is never one number across datasets}\\\ \text{out of scope this phase: ts, DAmP, CRISPRi}\end{gathered}$$"]:::lost
  DEC -.-> LOST

  %% declared last so dagre orders it at the edge of its rank rather than between
  %% the source box and the record, where it pushed the genotype channel off to one side
  DROP["$$\begin{gathered}\text{Hillenmeyer HOM dropped}\\\ 1.09\text{M records, one chemical}\\\ \text{neighbor above }0.5\end{gathered}$$"]:::lost
  SRC -.-> DROP
```

**What the diagram asserts, and the evidence.**

- The genotype channel is lossless for these datasets on the dosage axis, because every
  perturbation is either a full deletion or a one-of-two engineered copy-number variant
  (`audit_ploidy_representation.py`, read from the built stores).
- Ploidy needs no separate feature because the vector's background level is the ploidy, so a
  haploid deletion at 0 against a background of 1 stays distinguishable from a homozygous diploid
  deletion at 0 against a background of 2.
- The operator scaling reproduces the present model exactly when $\gamma(1) = 1$, since every
  record in every current dataset is a deletion with $m_j = 1$. What $\gamma(\tfrac{1}{2})$ should
  be is learned from Hoepfner's 648,977 gene-by-compound cells measured in both arms.
- Dose enters as a basis token and a masked log molar value rather than one number, because the
  three regimes are not convertible (`environment_axis_coverage.py`).
- The physical fields are constant within every dataset: temperature is stated for 1 to 2 percent
  of Hillenmeyer records and is 30 C everywhere else, only Vanacloig is anaerobic, pH is stated
  only in Vanacloig and 2 percent of Hillenmeyer HOM (`physical_axis_coverage.py`). A channel for
  them would be the dataset token under another name.
- The medium and the metabolite branch share the compound encoder because 30 of 37 media
  components are Yeast9 metabolites (`yeast9_molecule_coverage.py`).
- The dataset token at the readout is the mechanism smoke-tested in experiment 030 (GilaHyper job
  2867): a synthetic per-source offset of 0.3 was recovered as 0.285.
