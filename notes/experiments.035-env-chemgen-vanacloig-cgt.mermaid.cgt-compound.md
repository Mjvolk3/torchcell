---
id: 53fw0xcl06gzjrwav57czg5
title: Cgt Compound
desc: ''
updated: 1790810131906
created: 1790810131906
---

## 2026.09.30 - How the compound meets the genome in the 035 models

Two diagrams, drawn from the code as it ran. The first is the model that combines the
cell graph transformer with the molecular encoder (`train_factorized.py`,
`gene_encoder: cgt`, configs `r2_cgt.yaml`; slurm 3038). The second is the family of
compound mixings in `train_hit.py` (rounds 6 and 7), where the gene side is a lookup
table with SAGEConv message passing and the four `mix` options decide where the molecule
enters. Palette follows [[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]]:
purple input, orange embeddings and compound encoder, red transformer and output, yellow
loss and regularization.

Render with `bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.035-env-chemgen-vanacloig-cgt.mermaid.cgt-compound.md`
(two diagrams, so the outputs are `...cgt-compound-1.pdf` and `...cgt-compound-2.pdf`).

### Diagram 1: cell graph transformer as the gene encoder of the factorized model

Per training step a batch of $B$ strains is encoded once each (one transformer pass over
the wildtype graph, then the deletion operator per strain), and every strain is scored
against all $C$ training compounds at once, so the loss is over a $B \times C$ block of
the response matrix. Measured result: median centered Spearman 0.061 over 123 evaluations
against 0.287 for the lookup table in the same trainer, and 0.030 with the identity skip.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph Inputs["$$\text{Inputs}$$"]
    direction TB
    Strain["$$\begin{gathered}\text{Strain } s\ (3{,}598)\\\ S_s = \{g_s\} \cup \{\mathrm{PDR1}, \mathrm{PDR3}, \mathrm{SNQ2}\}\\\ \text{four deletions, as cell-graph indices}\end{gathered}$$"]
    Compound["$$\begin{gathered}\text{Compound } c\ (41)\\\ x_c \in \mathbb{N}^{2048}\ \text{FCFP4 counts}\end{gathered}$$"]
    CellGraph["$$\begin{gathered}\text{Wildtype cell graph}\\\ N = 6{,}607\ \text{genes},\ 9\ \text{graphs}\end{gathered}$$"]
  end

  subgraph GeneEncoder["$$\text{Gene encoder: cell graph transformer (shared across strains)}$$"]
    direction TB
    Table["$$\begin{gathered}\text{Learnable gene table}\\\ E \in \mathbb{R}^{N \times 180},\ \text{no precomputed features}\\\ \to\ \text{2-layer preprocessor}\end{gathered}$$"]
    Transformer["$$\begin{gathered}8\ \text{transformer layers},\ 9\ \text{heads},\ d = 180\\\ [\mathrm{CLS}] + N\ \text{gene tokens}\\\ H_{\mathrm{genes}} = T^{(8)} \circ \cdots \circ T^{(1)}(E)\ \in \mathbb{R}^{N \times 180}\end{gathered}$$"]
    GraphReg["$$\begin{gathered}\text{Graph regularization, layer 1}\\\ \mathcal{L}_{\mathrm{graph}} = \sum_g \mathrm{KL}(A_g \,\|\, \alpha^{(1,k_g)}),\ \lambda = 1\\\ \text{one head per graph}\end{gathered}$$"]
    DeleteOp["$$\begin{gathered}\text{Deletion operator (Type I)}\\\ \text{cross-attention from every gene onto } S_s\\\ H_{\mathrm{pert}} \in \mathbb{R}^{B \times N \times 180}\end{gathered}$$"]
    Gather["$$\begin{gathered}\text{Gather the deleted rows and sum}\\\ \tilde z_s = \sum_{g \in S_s} H_{\mathrm{pert}}[s, g]\ \in \mathbb{R}^{180}\\\ \text{identity skip: } \big\Vert\ \sum_{g \in S_s} E[g]\end{gathered}$$"]
    Project["$$\begin{gathered}\mathrm{LayerNorm} \to \mathrm{Linear}\\\ z_s \in \mathbb{R}^{64}\end{gathered}$$"]
  end

  subgraph CompoundEncoder["$$\text{Compound encoder (inductive: any molecule with a fingerprint)}$$"]
    direction TB
    Prep["$$\begin{gathered}\log(1 + x_c)\\\ \text{standardize per bit on the fold's training compounds}\end{gathered}$$"]
    MLP["$$\begin{gathered}\mathrm{Dropout}\,0.2 \to \mathrm{Linear}(2048 \to 256) \to \mathrm{GELU}\\\ \to \mathrm{Dropout} \to \mathrm{Linear}(256 \to 64)\\\ u_c \in \mathbb{R}^{64}\end{gathered}$$"]
  end

  subgraph Head["$$\text{Interaction head}$$"]
    direction TB
    Bilinear["$$\begin{gathered}\hat y_{sc} = b_{g_s} + w^{\top} u_c + \dfrac{z_s^{\top} u_c}{\sqrt{64}}\\\ \text{(gene bias + compound offset + bilinear)}\\\ \text{mlp option: } \mathrm{MLP}([z_s, u_c, z_s \odot u_c])\end{gathered}$$"]
  end

  subgraph Loss["$$\text{Loss and selection}$$"]
    direction TB
    MSE["$$\begin{gathered}\text{masked MSE on the standardized}\\\ \log_2 \dfrac{\mathrm{CPM}_{sc} + 1}{\overline{\mathrm{CPM}}_{s,\mathrm{ctrl}} + 1}\ \text{over the } B \times C_{\mathrm{train}} \text{ block}\\\ + \lambda\, \mathcal{L}_{\mathrm{graph}}\end{gathered}$$"]
    Select["$$\begin{gathered}\text{step picked by centered Spearman}\\\ \text{on 4 validation compounds; test = 8\text{-}9 held-out}\\\ \text{compounds, centered by the non-test mean}\end{gathered}$$"]
  end

  CellGraph --> Table
  CellGraph --> GraphReg
  Table --> Transformer
  GraphReg -.->|"$$\text{regularize}$$"| Transformer
  Transformer --> DeleteOp
  Strain --> DeleteOp
  DeleteOp --> Gather
  Table -.->|"$$\text{skip}$$"| Gather
  Gather --> Project
  Compound --> Prep
  Prep --> MLP
  Project --> Bilinear
  MLP --> Bilinear
  Bilinear --> MSE
  GraphReg --> MSE
  MSE --> Select

  classDef input fill:#E1D5E7,stroke:#846592,color:#1a1a1a
  classDef embed fill:#FFE6CC,stroke:#BD8800,color:#1a1a1a
  classDef trans fill:#F8CECC,stroke:#A24A46,color:#1a1a1a
  classDef reg fill:#FFF2CC,stroke:#BCA04C,color:#1a1a1a
  class Strain,Compound,CellGraph input
  class Table,Prep,MLP,Gather,Project embed
  class Transformer,DeleteOp,Bilinear trans
  class GraphReg,MSE,Select reg
```

### Diagram 2: the four compound mixings of `train_hit.py`

Same compound encoder and same loss. The gene side is a table over all 6,607 cell-graph
genes with `gcn_layers` SAGEConv layers over the union of the nine graphs (3,028,142
undirected edges; `relational` uses one SAGEConv per graph, averaged). `mix` picks where
the molecule enters. Measured (round 6 and 7, two message-passing layers, ten seeds):
`gene_attend` -0.006, `readout` -0.009, `hit_prop` -0.010 against nested ridge, all
intervals through zero; without message passing every mixing is 0.02 to 0.03 below ridge.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph Inputs2["$$\text{Inputs}$$"]
    direction TB
    Gene2["$$\begin{gathered}\text{Deleted gene } g_s\\\ \text{(the host deletions are constant, so dropped)}\end{gathered}$$"]
    Compound2["$$\begin{gathered}\text{Compound } c\\\ x_c \in \mathbb{N}^{2048}\end{gathered}$$"]
    Graphs2["$$\begin{gathered}\text{9 gene graphs}\\\ \text{union: } 3{,}028{,}142\ \text{edges}\\\ \text{row-normalized } \hat A_g \text{ per graph for propagation}\end{gathered}$$"]
  end

  subgraph GeneSide["$$\text{Gene side}$$"]
    direction TB
    Table2["$$\begin{gathered}\text{Gene table } T \in \mathbb{R}^{6607 \times 64}\end{gathered}$$"]
    Sage["$$\begin{gathered}L\ \text{SAGEConv layers over the union}\\\ h \leftarrow \mathrm{LN}\big(h + \mathrm{GELU}(\mathrm{SAGE}(h, A_{\cup}))\big)\\\ H \in \mathbb{R}^{N \times 64},\ h_s = H[g_s]\end{gathered}$$"]
  end

  subgraph CompoundSide["$$\text{Compound side}$$"]
    direction TB
    Enc2["$$\begin{gathered}\log(1+x_c) \to \text{standardize} \to \mathrm{MLP}(2048 \to 256 \to 64)\\\ u_c \in \mathbb{R}^{64}\end{gathered}$$"]
  end

  subgraph Mix["$$\text{Where the molecule enters (one of four)}$$"]
    direction TB
    Readout["$$\begin{gathered}\texttt{readout}\\\ [\,h_s,\ u_c,\ h_s \odot W u_c\,]\end{gathered}$$"]
    Hit["$$\begin{gathered}\texttt{hit}: \text{compound attends over all } N \text{ genes}\\\ a_c = \mathrm{softmax}_g\!\left(\dfrac{q(u_c)\, k(H)^{\top}}{\sqrt{d_h}\,\tau}\right) \in \Delta^{N},\ 8\ \text{heads}\\\ m_c = \sum_g a_{cg}\, v(H_g)\ \ (\text{hit vector})\\\ [\,h_s,\ u_c,\ h_s \odot W u_c,\ m_c,\ W_1 \log(1 + N a_{c g_s})\,]\end{gathered}$$"]
    HitProp["$$\begin{gathered}\texttt{hit\_prop}: \text{hit mass propagated, ego-net style}\\\ r^{(k)}_{c,g} = \big(\hat A_g^{\top}\big)^{k} a_c,\ k = 1..\text{hops},\ \text{per graph}\\\ \text{reach at the deleted gene: } \log(1 + N\, r_{c, g_s}) \in \mathbb{R}^{9\cdot\text{hops}+1}\\\ [\,\ldots \text{hit features} \ldots,\ \mathrm{MLP}(\text{reach})\,]\end{gathered}$$"]
    GeneAttend["$$\begin{gathered}\texttt{gene\_attend}: \text{deleted gene attends over}\\\ \{h_s,\ u_c,\ \mathbf{0}\ \text{null sink with learned logit bias}\}\\\ o_{sc} = \mathrm{LN}\big(h_s + W_o\,\mathrm{Attn}(h_s \to \{h_s, u_c, \mathbf 0\})\big)\\\ [\,o_{sc},\ u_c\,]\end{gathered}$$"]
  end

  subgraph Out2["$$\text{Head and loss}$$"]
    direction TB
    Head2["$$\begin{gathered}\hat y_{sc} = b_{g_s} + w^{\top} u_c + \mathrm{MLP}(\text{concat})\\\ \text{masked MSE on the standardized } \log_2 \text{ ratio}\end{gathered}$$"]
  end

  Table2 --> Sage
  Graphs2 --> Sage
  Gene2 --> Sage
  Compound2 --> Enc2
  Sage --> Readout
  Sage --> Hit
  Sage --> HitProp
  Sage --> GeneAttend
  Enc2 --> Readout
  Enc2 --> Hit
  Enc2 --> HitProp
  Enc2 --> GeneAttend
  Graphs2 -.->|"$$\hat A_g$$"| HitProp
  Readout --> Head2
  Hit --> Head2
  HitProp --> Head2
  GeneAttend --> Head2

  classDef input fill:#E1D5E7,stroke:#846592,color:#1a1a1a
  classDef embed fill:#FFE6CC,stroke:#BD8800,color:#1a1a1a
  classDef trans fill:#F8CECC,stroke:#A24A46,color:#1a1a1a
  classDef reg fill:#FFF2CC,stroke:#BCA04C,color:#1a1a1a
  class Gene2,Compound2,Graphs2 input
  class Table2,Enc2 embed
  class Sage,Readout,Hit,HitProp,GeneAttend trans
  class Head2 reg
```

What the two diagrams make visible side by side: in diagram 1 the molecule never touches
the graph. It meets the strain only at the bilinear head, after the transformer has
produced a strain vector with no knowledge of which compound is present, so the model is
a low-rank factorization $z_s^{\top} u_c$ whose gene factor happens to be computed by a
transformer. Diagram 2 is where the molecule is allowed to hit genes (`hit`), spread over
the network (`hit_prop`), or be attended to by the deleted gene (`gene_attend`); with two
message-passing layers all three reach ridge and none beats it.
