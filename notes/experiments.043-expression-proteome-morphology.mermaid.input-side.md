---
id: tej9irwr2vuz6ir6oqc6bmw
title: Input Side
desc: ''
updated: 1792449158662
created: 1792449158662
---

## 2026.10.19 - The input side of the 043 model, top to bottom

Companion to the `input-design.tex` section of `notes-tex/multimodal/043-datasets-primer/`
and to the prototype [[torchcell.models.genotype_tokens_043]]. Genotype classes become
change-type marks; the sequence encoder gives every reference gene one token; marks and
strain tokens feed the perturbation operator; the readouts come off the perturbed CLS and
gene states. Counts per record are the store census
(`experiments/043-expression-proteome-morphology/results/store_census.json`, Caudal means
for the isolate row). Palette as in
[[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]]: purple = input
and genotype, orange = encoders and tokens, red = operator and perturbed graph, yellow =
marks and strain-level tokens.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph Genotypes["$$\text{Genotype classes in the 043 store}$$"]
    direction LR
    Del["$$\begin{gathered}\text{Deletion, double deletion}\\\ \texttt{DeletionPerturbation}\\\ M = 1\ \text{or}\ 2\end{gathered}$$"]
    Iso["$$\begin{gathered}\text{Natural isolate, median per record}\\\ \text{variant}\ 5{,}069,\ \text{LoF}\ 362\\\ \text{absent}\ 126,\ \text{present}\ 78\end{gathered}$$"]
    Seg["$$\begin{gathered}\text{Segregant}\\\ \texttt{HaplotypeBlock}\ \{\chi, \mathrm{start}, \mathrm{end}, \mathrm{parent}, \mathrm{post}\}\\\ \text{no perturbation list}\end{gathered}$$"]
    Prom["$$\begin{gathered}\text{Promoter variant}\\\ p \in \Sigma^{80}\ \text{in pTpA, YFP reporter}\\\ 71{,}158\ \text{records}\end{gathered}$$"]
  end

  subgraph Marks["$$\text{Change-type marks per gene}\quad c \in \mathcal{C}^{|G|},\ m_i = \mathbf{1}[c_i \neq \mathrm{ref}]$$"]
    direction LR
    Vocab["$$\begin{gathered}\mathcal{C} = \{\mathrm{ref}, \mathrm{del}, \mathrm{var}, \mathrm{lof}, \mathrm{abs}, \mathrm{pres}, \mathrm{prom}\}\\\ \text{precedence: abs} \succ \text{del} \succ \text{lof} \succ \text{var}\end{gathered}$$"]
    SegRule["$$\begin{gathered}\text{definite assumption}\\\ c_i = \mathrm{var},\ a_i = \text{non-ref parent}\\\ \text{if block posterior} \geq 0.9\ \text{contains gene}\ i\\\ \text{parent's real allele token}\end{gathered}$$"]
  end

  subgraph Encoders["$$\text{Composite sequence token, four frozen encoders}$$"]
    direction LR
    SeqEnc["$$\begin{gathered}f_k : \Sigma^* \to \mathbb{R}^{d_k},\ \text{per-block LN and}\ W_k\\\ \text{FUDT upstream 1{,}003 bp (768)}\\\ \text{FUDT downstream 300 bp (768)}\\\ \text{ProtT5-XL (1{,}024), CaLM (768)}\\\ \text{on the realized allele}\end{gathered}$$"]
    Pos["$$\begin{gathered}\text{position}\ P_i\\\ \kappa_\chi C_\chi + \mathrm{Fourier}(\rho_i) W_\rho\\\ +\, w_c \log_2 \kappa_i\end{gathered}$$"]
  end

  subgraph Tokens["$$\text{Reference contexts and gene tokens}$$"]
    direction TB
    Ctx["$$\begin{gathered}\text{reference contexts}\ s\\\ \text{S288C, 979 isolates, 5 parents}\\\ \text{encoded once, cached}\end{gathered}$$"]
    Xref["$$\begin{gathered}X^{(s)}_i = \textstyle\sum_k W_k \mathrm{LN}_k(f_k(s_i^k)) + P_i\\\ X^{(s)} \in \mathbb{R}^{|G^{(s)}| \times d}\end{gathered}$$"]
    Xstrain["$$\begin{gathered}\tilde X_{b,i} = X_i + T_{c_{b,i}} + A_{a_{b,i}}\\\ \tilde X \in \mathbb{R}^{B \times |G| \times d}\end{gathered}$$"]
    Urep["$$\begin{gathered}u_{\mathrm{rep}}:\ f_{\mathrm{up}}\ \text{on the construct window}\\\ +\, \text{YFP blocks} + T_{\mathrm{prom}}\\\ \text{extra token, off graph}\end{gathered}$$"]
  end

  subgraph Encoder["$$\text{Reference encoder, once per context}$$"]
    Enc["$$\begin{gathered}E = \mathrm{Enc}([x_{\mathrm{CLS}}; X^{(s)}])\\\ = [h; h_1; \dots; h_{|G|}] \in \mathbb{R}^{(|G|+1) \times d}\end{gathered}$$"]
  end

  subgraph Operator["$$\text{Perturbation operator, per strain}$$"]
    direction TB
    PTok["$$\begin{gathered}\text{keys and values: engineered edits}\\\ \mathbf{p}_t = h_{g(t)} + T_{c} + A_{a},\ m_{g(t)} = 1\end{gathered}$$"]
    Attn["$$\begin{gathered}s_{i,t} = q_i \cdot k_t / \sqrt{d_k},\ \Delta h_i = \sum_t \beta_{i,t} v_t\\\ h_i^{\mathrm{pert}} = \mathrm{LN}(\hat h_i + \mathrm{FFN}(\hat h_i))\\\ \beta:\ \text{softmax, or softmax with null sink}\end{gathered}$$"]
  end

  subgraph Perturbed["$$\text{Perturbed cell graph, read by the heads}$$"]
    direction LR
    CLS["$$\begin{gathered}h^{\mathrm{pert}} \in \mathbb{R}^{B \times d}\\\ \text{record-level head}\\\ [h^{\mathrm{pert}} \| \mathrm{mean}_i H_i \| z_S]\end{gathered}$$"]
    Hg["$$\begin{gathered}H \in \mathbb{R}^{B \times |G| \times d}\\\ \text{gene-level head}\\\ z_S = \mathrm{pool}\{H_i : m_i = 1\}\end{gathered}$$"]
    HeadTok["$$\begin{gathered}\text{at the head only}\\\ t_{\mathrm{type}},\ t_{\mathrm{env}},\ t_{\mathrm{data}}\\\ \text{dataset never in the trunk}\end{gathered}$$"]
  end

  Genotypes ~~~ Marks
  Marks ~~~ Tokens
  Del --> Vocab
  Iso --> Vocab
  Seg --> SegRule
  Prom --> Vocab
  SegRule --> Xstrain
  Vocab --> Xstrain
  SeqEnc --> Xref
  Pos --> Xref
  Ctx --> Xref
  Xref --> Xstrain
  Xref --> Enc
  Prom --> Urep
  Enc --> PTok
  Xstrain -.->|"$$T_c + A_a\ \text{on the marked rows}$$"| PTok
  Urep --> PTok
  PTok --> Attn
  Enc --> Attn
  Attn --> CLS
  Attn --> Hg
  HeadTok -.-> Hg

  classDef input fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef marks fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef encoder fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef operator fill:#F8CECC,stroke:#A24A46,stroke-width:2px

  class Del,Iso,Seg,Prom input
  class Vocab,SegRule,Pos,HeadTok marks
  class SeqEnc,Ctx,Xref,Xstrain,Urep,Enc encoder
  class PTok,Attn,CLS,Hg operator
```

Regenerate the figure asset (name matches this note):

```bash
bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.043-expression-proteome-morphology.mermaid.input-side.md
# -> notes/assets/pdf-output/experiments.043-expression-proteome-morphology.mermaid.input-side.pdf
```

## 2026.10.20 - Review pass: composite token, reference contexts, head-only context tokens

The diagram above was redrawn for the 2026-10-20 review pass
([[experiments.043-expression-proteome-morphology]], "Review pass"). The sequence-encoder
box now lists the four frozen encoders of the composite token (FUDT upstream and
downstream, ProtT5-XL, CaLM; one-hot is gone and survives only as an ablation arm); a
reference-context box feeds the gene tokens (S288C, 979 isolates, five non-BY parents,
each encoded once and cached); the reporter's promoter block is the FUDT upstream encoder
on the construct window, so the separate promoter encoder is gone; the strain-token box is
gone, the allele parent rides on the segregant rule, and the type, environment and dataset
tokens enter at the heads only. Isolate counts are medians per Caudal record from
`results/data_shapes_043.json`.
