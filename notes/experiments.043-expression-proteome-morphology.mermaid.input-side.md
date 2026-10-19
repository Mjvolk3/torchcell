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
    Iso["$$\begin{gathered}\text{Natural isolate}\\\ \text{variant}\ 5{,}047,\ \text{LoF}\ 383\\\ \text{absent}\ 135,\ \text{present}\ 84\end{gathered}$$"]
    Seg["$$\begin{gathered}\text{Segregant}\\\ \texttt{HaplotypeBlock}\ \{\chi, \mathrm{start}, \mathrm{end}, \mathrm{parent}, \mathrm{post}\}\\\ \text{no perturbation list}\end{gathered}$$"]
    Prom["$$\begin{gathered}\text{Promoter variant}\\\ p \in \Sigma^{80}\ \text{in pTpA, YFP reporter}\\\ 71{,}158\ \text{records}\end{gathered}$$"]
  end

  subgraph Marks["$$\text{Change-type marks per gene}\quad c \in \mathcal{C}^{|G|},\ m_i = \mathbf{1}[c_i \neq \mathrm{ref}]$$"]
    direction LR
    Vocab["$$\begin{gathered}\mathcal{C} = \{\mathrm{ref}, \mathrm{del}, \mathrm{var}, \mathrm{lof}, \mathrm{abs}, \mathrm{pres}, \mathrm{prom}\}\\\ \text{precedence: abs} \succ \text{del} \succ \text{lof} \succ \text{var}\end{gathered}$$"]
    SegRule["$$\begin{gathered}\text{definite assumption}\\\ c_i = \mathrm{var},\ a_i = \text{non-ref parent}\\\ \text{if block posterior} \geq 0.9\ \text{contains gene}\ i\end{gathered}$$"]
    StrainTok["$$\begin{gathered}\text{strain tokens}\ u_k\\\ \text{parent, medium, dataset}\end{gathered}$$"]
  end

  subgraph Encoders["$$\text{Sequence encoders, swappable}$$"]
    direction LR
    SeqEnc["$$\begin{gathered}f_\theta : \Sigma^* \to \mathbb{R}^{d_f}\\\ \text{ProtT5, NT windows, FUDT, CALM, one-hot}\\\ \text{on}\ s_i(\gamma_{\mathrm{ref}});\ \text{later on the variant allele}\end{gathered}$$"]
    PromEnc["$$\begin{gathered}g_\phi : \Sigma^* \to \mathbb{R}^{d_g}\\\ \text{trained on Vaishnav}\end{gathered}$$"]
    Pos["$$\begin{gathered}\text{position (optional)}\\\ \pi_i = (\chi_i, \rho_i)\end{gathered}$$"]
  end

  subgraph Tokens["$$\text{Gene tokens and strain tokens}$$"]
    direction TB
    Xref["$$\begin{gathered}X = W_f f_\theta(s) + C_\chi + w_\rho \rho\\\ X \in \mathbb{R}^{|G| \times d},\ \text{shared}\end{gathered}$$"]
    Xstrain["$$\begin{gathered}\tilde X_{b,i} = X_i + T_{c_{b,i}} + A_{a_{b,i}}\\\ \tilde X \in \mathbb{R}^{B \times |G| \times d}\end{gathered}$$"]
    Urep["$$\begin{gathered}u_{\mathrm{rep}} = W_g g_\phi(p) + T_{\mathrm{prom}} + X_{\mathrm{YFP}}\\\ \text{extra token, off graph}\end{gathered}$$"]
  end

  subgraph Encoder["$$\text{Reference encoder, once per batch}$$"]
    Enc["$$\begin{gathered}E = \mathrm{Enc}([x_{\mathrm{CLS}}; X])\\\ = [h; h_1; \dots; h_{|G|}] \in \mathbb{R}^{(|G|+1) \times d}\end{gathered}$$"]
  end

  subgraph Operator["$$\text{Perturbation operator, per strain}$$"]
    direction TB
    PTok["$$\begin{gathered}\text{keys and values}\\\ \mathbf{p}_t = h_{g(t)} + T_{c} + A_{a},\ m_{g(t)} = 1\\\ \cup\ \{u_k\}\end{gathered}$$"]
    Attn["$$\begin{gathered}s_{i,t} = q_i \cdot k_t / \sqrt{d_k},\ \Delta h_i = \sum_t \beta_{i,t} v_t\\\ h_i^{\mathrm{pert}} = \mathrm{LN}(\hat h_i + \mathrm{FFN}(\hat h_i))\\\ \beta:\ \text{softmax, or softmax with null sink}\end{gathered}$$"]
  end

  subgraph Perturbed["$$\text{Perturbed cell graph}$$"]
    direction LR
    CLS["$$\begin{gathered}h^{\mathrm{pert}} \in \mathbb{R}^{B \times d}\\\ \text{CLS, global readouts}\\\ [h^{\mathrm{pert}} \| \mathrm{mean}_i H_i \| z_S]\end{gathered}$$"]
    Hg["$$\begin{gathered}H \in \mathbb{R}^{B \times |G| \times d}\\\ \text{gene states, per-gene readouts}\\\ z_S = \mathrm{pool}\{H_i : m_i = 1\}\end{gathered}$$"]
  end

  Del --> Vocab
  Iso --> Vocab
  Seg --> SegRule
  Prom --> Vocab
  Seg --> StrainTok
  SegRule --> Xstrain
  Vocab --> Xstrain
  SeqEnc --> Xref
  Pos --> Xref
  Xref --> Xstrain
  Xref --> Enc
  PromEnc --> Urep
  Prom --> PromEnc
  Enc --> PTok
  Xstrain -.->|"$$T_c + A_a\ \text{on the marked rows}$$"| PTok
  StrainTok --> PTok
  Urep --> PTok
  PTok --> Attn
  Enc --> Attn
  Attn --> CLS
  Attn --> Hg

  classDef input fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef marks fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef encoder fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef operator fill:#F8CECC,stroke:#A24A46,stroke-width:2px

  class Del,Iso,Seg,Prom input
  class Vocab,SegRule,StrainTok,Pos marks
  class SeqEnc,PromEnc,Xref,Xstrain,Urep,Enc encoder
  class PTok,Attn,CLS,Hg operator
```

Regenerate the figure asset (name matches this note):

```bash
bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.043-expression-proteome-morphology.mermaid.input-side.md
# -> notes/assets/pdf-output/experiments.043-expression-proteome-morphology.mermaid.input-side.pdf
```
