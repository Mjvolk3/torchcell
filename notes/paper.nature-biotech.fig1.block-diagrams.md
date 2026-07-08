---
id: glptqd7k4x37ygqd9lqrs0x
title: Block Diagrams
desc: ''
updated: 1783460804368
created: 1783460804368
---

Block-diagrams for **Fig 1 panels d–f** of the CGT paper, drawn in the
Set-Transformer block style (self-attention block SAB, and the base cross-attention
block MAB; Lee et al. 2019) the co-authors shared. Companion to the general-case math
in [[paper.nature-biotech.fig1.perturbation-operator]] and the [[paper.methods]]
equations. The mermaid blocks are draw.io-ready sketches; render to a vector with
`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/paper.nature-biotech.fig1.block-diagrams.md`.

## 2026.06.29 - CGT as Set-Transformer blocks (encoder · perturbation operator · graph regularization)

### Block vocabulary

The CGT reuses two Set-Transformer attention primitives, specialised to the cell; the
readouts are plain mean-pool + MLP (no learned pooling block):

| panel    | component                               | block           | what it is                                        |
|----------|-----------------------------------------|-----------------|---------------------------------------------------|
| d        | cell encoder $F_\theta$                 | **SAB ×L**      | self-attention over gene tokens                   |
| e        | perturbation operator $\mathcal T_\psi$ | **cross-attn (CAB)**   | cross-attention: cells query the perturbation set |
| readouts | invariant heads                         | **mean-pool + MLP** | average gene rows, then a per-task MLP        |
| f        | graph regularization                    | *(not a block)* | a training-time KL on a SAB attention head        |

**Caution — soft, not masked.** The "Masked Attention Block (edge mask)" is the *hard*
graph-convolution we reject. Our encoder is an **unmasked SAB**; the graph enters only as a
**soft KL target** (panel f), so data can overrule a wrong or untested edge. Do not draw an
edge mask in the encoder. We also avoid the label "MAB" for the operator: in the
Set-Transformer paper MAB is the *Multihead* Attention Block (the base cross-attention block,
with $\mathrm{SAB}(X)=\mathrm{MAB}(X,X)$), whereas the shared figure relabels MAB as a
*Masked* Attention Block — a different thing. Our operator is cross-attention and is **never
masked**, so we call it a cross-attention block (CAB). (The three perturbation operations — del / OE / KD — are not three
operators here; they are the three values of the type $\tau$, carried by the token
embedding $\mathbf t_\tau$ into a single $\mathcal T_\psi$.)

### d — Cell encoder: SAB ×L

$$H^{(\ell)}=\mathrm{SAB}\big(H^{(\ell-1)}\big),\qquad H^{(0)}=\mathrm{Embed}(G),\qquad H=H^{(L)}.$$

Pre-norm self-attention, then MLP, each with a residual ([[paper.methods]] eq. qkv / attn /
layer). Selected heads $\alpha^{(\ell,a)}$ are softly aligned to graphs (panel f); no mask.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph LR
  X["$$H^{(\ell-1)}$$"]:::io
  N1["$$\text{Norm}$$"]:::norm
  A["$$\begin{gathered}\text{self-attention}\\\ Q,K,V=H^{(\ell-1)}\\\ \text{heads }\alpha^{(\ell,a)}\end{gathered}$$"]:::attn
  P1(("$$+$$")):::add
  N2["$$\text{Norm}$$"]:::norm
  M["$$\text{FFN}$$"]:::mlp
  P2(("$$+$$")):::add
  Y["$$H^{(\ell)}$$"]:::io
  X --> N1 --> A --> P1 --> N2 --> M --> P2 --> Y
  X -.->|"$$\text{skip}$$"| P1
  P1 -.->|"$$\text{skip}$$"| P2
  classDef io fill:#FFE6CC,stroke:#BD8800,color:#000
  classDef norm fill:#F5EEDD,stroke:#B7AC93,color:#000
  classDef attn fill:#F8CECC,stroke:#A24A46,color:#000
  classDef mlp fill:#FFF2CC,stroke:#BCA04C,color:#000
  classDef add fill:#ffffff,stroke:#333333,color:#000
```

### e — Perturbation operator: cross-attention block (CAB)

The perturbation operation: every gene (a query) attends to the perturbation context (the
keys and values), a transformation inside embedding space. Build the keys/values from the
perturbation set, then cross-attend with a residual:

$$
\mathbf{p}_t=r(e_t)+\mathbf t_{\tau_t}+m_t\,\mathbf w,\qquad \mathbf{P}=(\mathbf{p}_1,\dots,\mathbf{p}_M),
$$
$$
\beta_{i,t}=\operatorname*{softmax}_{t\in[M]}\frac{(h_iW_Q)\cdot(\mathbf{p}_tW_K)}{\sqrt{d_k}},\qquad
\Delta h_i=\sum_{t=1}^{M}\beta_{i,t}\,(\mathbf{p}_tW_V),\qquad
H_{\mathrm{pert}}=\mathcal T_\psi(H,p).
$$

Many queries (all $N$ genes), few keys ($\lvert p\rvert$); output is equivariant and feeds
the readouts; $\mathcal T_\psi$ generalises from gene tokens to drug / environment tokens by widening $p$.
Full treatment (properties, cartoon, generalisation): [[paper.nature-biotech.fig1.perturbation-operator]].

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph LR
  PP["$$\begin{gathered}\text{perturbation}\\\ p=\{(e_t,\tau_t,m_t)\}\end{gathered}$$"]:::pert
  C["$$\begin{gathered}\mathbf{p}_t=r(e_t)+\mathbf{t}_\tau+m_t\,\mathbf{w}\\\ r(e_t)=h_j\ (\text{gene})\ \text{or}\ \mathrm{Embed}(e_t)\ (\text{drug/env})\\\ m_t\ (\text{scalar})\times\mathbf{w}\ (\text{direction})\\\ \mathbf{P}=(\mathbf{p}_1,\dots,\mathbf{p}_M)\ \ \text{keys, values}\end{gathered}$$"]:::pert
  H["$$\begin{gathered}H\\\ \text{all }N\text{ genes}=\text{queries}\end{gathered}$$"]:::io
  N1["$$\text{Norm}$$"]:::norm
  A["$$\begin{gathered}\text{cross-attention}\\\ Q=H,\ \ K,V=\mathbf{P}\\\ \beta_{i,t}=\operatorname*{softmax}_{t}\dfrac{q_i\cdot k_t}{\sqrt{d_k}}\end{gathered}$$"]:::attn
  P1(("$$+$$")):::add
  N2["$$\text{Norm}$$"]:::norm
  M["$$\text{FFN}$$"]:::mlp
  P2(("$$+$$")):::add
  Y["$$\begin{gathered}H_{\mathrm{pert}}\\\ \text{equivariant}\end{gathered}$$"]:::io
  PP --> C --> A
  H --> N1 --> A --> P1 --> N2 --> M --> P2 --> Y
  H -.->|"$$\text{skip}$$"| P1
  P1 -.->|"$$\text{skip}$$"| P2
  classDef io fill:#FFE6CC,stroke:#BD8800,color:#000
  classDef pert fill:#E1D5E7,stroke:#846592,color:#000
  classDef norm fill:#F5EEDD,stroke:#B7AC93,color:#000
  classDef attn fill:#F8CECC,stroke:#A24A46,color:#000
  classDef mlp fill:#FFF2CC,stroke:#BCA04C,color:#000
  classDef add fill:#ffffff,stroke:#333333,color:#000
```

### f — Attention–graph regularization (graph → adjacency + attention matrix → regularize)

A gene graph becomes an $N\times N$ adjacency, the same shape as a gene–gene attention head,
so the two are aligned row by row with a KL penalty:

$$
\tilde A^{(k)}=(D^{(k)})^{-1}A^{(k)},\qquad
\Omega_k=\sum_{i\in\mathcal I_k}D_{\mathrm{KL}}\!\big(\tilde A^{(k)}_{i,:}\,\big\|\,\alpha^{(\ell_k,a_k)}_{i,:}\big),\qquad
\mathcal L=\mathcal L_y+\sum_{k=1}^{K}\lambda_k\,\Omega_k.
$$

One graph supervises one head. In experiment 010 these are $K=9$ gene–gene graphs (physical,
regulatory, TFLink, six STRING channels); metabolism is bipartite, has no $N\times N$
adjacency, and is excluded. The load-bearing point for the panel: $\tilde A^{(k)}$ and the
attention head are the **same $N\times N$ square**, so draw both as matching gene×gene
heatmaps with a KL between them.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph LR
  G["$$\begin{gathered}\text{gene network, relation }k\\\ \text{nodes = genes, edges = interactions}\end{gathered}$$"]:::pert
  Adj["$$\begin{gathered}\text{adjacency }A^{(k)}\\\ N\times N,\ \in\{0,1\}\end{gathered}$$"]:::io
  At["$$\begin{gathered}\text{row-normalize}\\\ \tilde{A}^{(k)}=(D^{(k)})^{-1}A^{(k)}\end{gathered}$$"]:::io
  SAB["$$\begin{gathered}\text{SAB head }\alpha^{(\ell_k,a_k)}\\\ N\times N\ \text{attention}\end{gathered}$$"]:::attn
  KL["$$D_{\mathrm{KL}}\big(\tilde{A}^{(k)}_{i,:}\,\big\|\,\alpha^{(\ell_k,a_k)}_{i,:}\big)=\Omega_k$$"]:::loss
  L["$$\begin{gathered}\mathcal{L}=\mathcal{L}_y+\sum_k\lambda_k\,\Omega_k\\\ K\text{ graphs}\to K\text{ heads}\end{gathered}$$"]:::loss
  G --> Adj --> At --> KL
  SAB --> KL --> L
  classDef io fill:#FFE6CC,stroke:#BD8800,color:#000
  classDef pert fill:#E1D5E7,stroke:#846592,color:#000
  classDef attn fill:#F8CECC,stroke:#A24A46,color:#000
  classDef loss fill:#FFF2CC,stroke:#BCA04C,color:#000
```

### Readouts — mean-pool + MLP (if shown in the panel)

Invariant heads **average** the gene rows and pass the mean through a per-task MLP: fitness
averages all $N$ genes, and the interaction head concatenates the whole-cell CLS token with
the mean of the perturbed-gene rows. Equivariant heads (expression, protein) are per-gene
MLPs; gene-set heads (morphology) average a fixed gene set $\mathcal G_s$. There is **no
learned pooling / seed query** (no PMA) — the pooling is a plain arithmetic mean
([[paper.methods]] eq. fit / int / expr / morph).

## 2026.07.01 - Perturbation operator: intuition panel (soft routing, replaces the CAB wiring)

The CAB wiring above (§e) shows the *plumbing* — but the operator's plumbing is just a
cross-attention block; its **point is the behavior**, which the wiring hides. Keep the SAB
**wiring** for the cell-encoder panel (self-attention *is* the honest picture there), but
draw the perturbation panel as the **soft-routing / spotlight** view below. This promotes the
A/B/C cartoon of [[paper.nature-biotech.fig1.perturbation-operator]] from an inset to the panel.

**What it shows (the four key features at a glance):**

- **Broadcast, few→many.** A *small* perturbation set $p$ (keys/values) is queried by *all*
  $N$ genes — the many-queries ← few-keys asymmetry.
- **Soft routing, not lookup.** Each gene pulls the effect **by relevance** $\beta_{i,t}$:
  genes close to the perturbed one in $H$ (functionally related) get large $\beta$ and their
  embedding **shifts**; unrelated genes get $\beta\approx0$ and stay put. The shift **is** the
  (genetic) interaction.
- **The update.** the per-gene update is the $\beta$-weighted blend of the token values
  $\mathbf{p}_tW_V$: $\Delta h_i=\sum_t\beta_{i,t}(\mathbf{p}_tW_V)$, added residually,
  $h_i^{\mathrm{pert}}=h_i+\Delta h_i$.
- **Generality.** $p$ holds gene tokens *and* drug/environment tokens — $e_t$ ranges over
  gene / drug / environment; a larger $p$, no new module.
- **Equivariant output.** one row per gene → $H_{\mathrm{pert}}$.

Uses the shared 5-node toy cell (delete $g_2$; its neighbour $g_3$ is *related* and shifts,
non-neighbour $g_5$ is *unrelated* and does not). Residual/FFN plumbing drops to a caption
("cross-attention block; full wiring in Methods / §e").

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph LR
  subgraph P["$$\begin{gathered}\text{perturbation } p=\{(e_t,\tau_t,m_t)\}_{t=1}^{M}\\\ \text{keys/values (few)}\end{gathered}$$"]
    direction TB
    d2["$$(g_2,\ \mathrm{del})$$"]:::pert
    df["$$e_t:\ \text{gene, drug, or environment}$$"]:::pertf
  end
  subgraph Q["$$\text{all } N \text{ genes — queries}$$"]
    direction TB
    g3(("$$g_3$$")):::gene
    g5(("$$g_5$$")):::gene
  end
  d2 ==>|"$$\beta_{3}\ \text{large (related)}$$"| g3
  d2 -.->|"$$\beta_{5}\approx 0\ (\text{unrelated})$$"| g5
  g3 --> s3["$$\Delta h_3=\textstyle\sum_t \beta_{3,t}\,(\mathbf{p}_tW_V)\ \Rightarrow\ h_3\ \text{shifts (interaction)}$$"]:::hot
  g5 --> s5["$$\Delta h_5\approx 0\ \Rightarrow\ h_5\ \text{unchanged}$$"]:::io
  s3 --> HP["$$\begin{gathered}H_{\mathrm{pert}}:\ h_i^{\mathrm{pert}}=h_i+\Delta h_i\\\ \text{per-gene, equivariant}\end{gathered}$$"]:::out
  s5 --> HP
  classDef pert fill:#E1D5E7,stroke:#846592,color:#000
  classDef pertf fill:#EFE7F3,stroke:#B0A0BE,color:#000
  classDef gene fill:#FFE6CC,stroke:#BD8800,color:#000
  classDef hot fill:#F8CECC,stroke:#A24A46,color:#000
  classDef io fill:#FFF2CC,stroke:#BCA04C,color:#000
  classDef out fill:#FFF2CC,stroke:#BCA04C,color:#000
```

### Related

- [[paper.nature-biotech.fig1.perturbation-operator]] — operator general case (gene / drug / env tokens).
- [[paper.nature-biotech.fig1.gat-cgt-equivalence]] — panel g GAT ↔ CGT equivalence + operator advantages.
- [[paper.methods]] — markdown ideation; canonical text in `paper/nature-biotech/sections/methods.tex`.
