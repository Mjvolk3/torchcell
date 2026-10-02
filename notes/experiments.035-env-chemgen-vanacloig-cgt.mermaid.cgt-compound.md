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

## 2026.10.02 - Diagram 3: the best model as it ran, and where the knockout enters

The best number of the experiment (median centered Spearman 0.334, +0.032 against nested
ridge, compound-level CI through zero) is a plain mean of nested ridge and the environment
encoder (`head: env_encoder`, `cgt_layers: 1`, `cgt_lambda: 0`, nine seeds). Drawn from
`EnvironmentEncoder` in `train_factorized.py`. Gray boxes are computed and then NOT used
by the prediction, or carry no strain information: the graph prior is off, so no graph
edge enters; the post-deletion states of the 6,603 remaining genes are discarded; the
pooled field and the compound token are the same for every strain under a compound. The
only strain-specific quantities are the four deleted genes' own rows, read twice.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph Inputs["$$\text{Inputs}$$"]
    direction TB
    Strain["$$\begin{gathered}\text{Strain } s\ (3{,}598\ \text{knockouts})\\\ S_s = \{g_s\} \cup \{\mathrm{PDR1}, \mathrm{PDR3}, \mathrm{SNQ2}\}\\\ \text{one screened deletion + three host deletions}\end{gathered}$$"]
    Compound["$$\begin{gathered}\text{Compound } c\ (41)\\\ x_c \in \mathbb{N}^{2048}\ \text{FCFP4 counts}\end{gathered}$$"]
    CellGraph["$$\begin{gathered}\text{Wildtype cell graph}\\\ N = 6{,}607\ \text{genes},\ 9\ \text{graphs}\\\ \text{used only to name the } N \text{ tokens}\end{gathered}$$"]
  end

  subgraph Ridge["$$\text{Nested ridge (no model of the knockout)}$$"]
    direction TB
    Kernel["$$\begin{gathered}\text{linear kernel on standardized } \log(1+x)\\\ K_{cc'} \text{ over the fitted compounds}\end{gathered}$$"]
    RidgePred["$$\begin{gathered}\hat y^{\mathrm{ridge}}_{sc} = \bar y_s + \sum_{c' \in \mathrm{fit}} \alpha_{cc'}\,(y_{sc'} - \bar y_s)\\\ \text{the gene is a row index; its profile over}\\\ \text{fitted compounds is interpolated}\end{gathered}$$"]
  end

  subgraph GeneEncoder["$$\text{Gene encoder: one transformer layer over a learned table}$$"]
    direction TB
    Table["$$\begin{gathered}\text{Learnable gene table}\\\ E \in \mathbb{R}^{N \times 180}\end{gathered}$$"]
    Transformer["$$\begin{gathered}1\ \text{self-attention layer},\ 9\ \text{heads},\ d = 180\\\ H = T(E) \in \mathbb{R}^{N \times 180}\ \text{(wildtype field)}\end{gathered}$$"]
    GraphReg["$$\begin{gathered}\text{Graph prior } \mathrm{KL}(A_g \,\|\, \alpha)\\\ \lambda = 0:\ \text{OFF in the best arm}\\\ \text{(1e-3 and 1 scored the same, round 8)}\end{gathered}$$"]
    DeleteOp["$$\begin{gathered}\text{Deletion operator (Type I)}\\\ H_{\mathrm{pert}}[s] \in \mathbb{R}^{N \times 180}:\ \text{every gene}\\\ \text{updated after } S_s \text{ is removed}\end{gathered}$$"]
    Hdel["$$\begin{gathered}\text{Read the deleted rows only}\\\ h^{\mathrm{del}}_s = \sum_{g \in S_s} H_{\mathrm{pert}}[s, g]\ \in \mathbb{R}^{180}\end{gathered}$$"]
    Discard["$$\begin{gathered}\text{Post-deletion state of the other}\\\ N - 4 = 6{,}603\ \text{genes: discarded}\end{gathered}$$"]
  end

  subgraph Env["$$\text{Environment encoder: the compound as a token the genes attend to}$$"]
    direction TB
    MLP["$$\begin{gathered}\mathrm{Dropout} \to \mathrm{Linear}(2048 \to 256) \to \mathrm{GELU}\\\ \to \mathrm{Dropout} \to \mathrm{Linear}(256 \to 180)\\\ e_c \in \mathbb{R}^{180}\end{gathered}$$"]
    Seq["$$\begin{gathered}\text{Sequence } [\,e_c\,;\, H\,] \in \mathbb{R}^{(N+1) \times 180}\\\ \text{built on the WILDTYPE field, same for every strain}\end{gathered}$$"]
    TokenLayer["$$\begin{gathered}1\ \text{pre-norm attention layer (flash)}\\\ H^{c} = \text{the cell in medium } c\ \in \mathbb{R}^{N \times 180}\\\ \text{(2 layers scored the same, round 14)}\end{gathered}$$"]
    Henv["$$\begin{gathered}\text{Read the deleted rows only}\\\ h^{\mathrm{env}}_{sc} = \sum_{g \in S_s} H^{c}[g]\ \in \mathbb{R}^{180}\\\ \text{the ONLY place the strain meets the compound}\end{gathered}$$"]
    Pooled["$$\begin{gathered}\bar h^{c} = \tfrac{1}{N}\sum_g H^{c}[g],\quad t^{c} = \text{compound token after the layer}\\\ \text{identical for every strain under } c\end{gathered}$$"]
  end

  subgraph Head["$$\text{Readout}$$"]
    direction TB
    Concat["$$\begin{gathered}\mathrm{LayerNorm}\big([\,h^{\mathrm{env}}_{sc}\,;\,h^{\mathrm{del}}_s\,;\,\bar h^{c}\,;\,t^{c}\,]\big) \in \mathbb{R}^{720}\\\ \to \mathrm{Linear}(720 \to 256) \to \mathrm{GELU} \to \mathrm{Dropout} \to \mathrm{Linear}(256 \to 1)\end{gathered}$$"]
    Pred["$$\begin{gathered}\hat y^{\mathrm{enc}}_{sc} = b_{g_s} + w^{\top} e_c + \mathrm{MLP}(\cdot)\\\ \text{gene bias + compound offset + interaction}\end{gathered}$$"]
  end

  subgraph Loss["$$\text{Training and the stack}$$"]
    direction TB
    MSE["$$\begin{gathered}\text{masked MSE on the standardized response over a}\\\ B = 128\ \text{strains} \times C_{\mathrm{fit}}\ \text{block; fit on all 32\text{-}33 non-test}\\\ \text{compounds, 50 epochs, last step kept, 9 seeds averaged}\end{gathered}$$"]
    Stack["$$\begin{gathered}\hat y_{sc} = \tfrac{1}{2}\big(\hat y^{\mathrm{ridge}}_{sc} + \hat y^{\mathrm{enc}}_{sc}\big)\\\ \text{scored per held-out compound: Spearman across strains}\\\ \text{after each side subtracts its fitted-compound mean}\end{gathered}$$"]
  end

  CellGraph --> Table
  CellGraph -.->|"$$\lambda = 0$$"| GraphReg
  GraphReg -.-> Transformer
  Table --> Transformer
  Transformer --> DeleteOp
  Strain --> DeleteOp
  DeleteOp --> Hdel
  DeleteOp -.-> Discard
  Transformer --> Seq
  Compound --> MLP
  MLP --> Seq
  Seq --> TokenLayer
  TokenLayer --> Henv
  Strain --> Henv
  TokenLayer --> Pooled
  Henv --> Concat
  Hdel --> Concat
  Pooled --> Concat
  Concat --> Pred
  MLP -->|"$$w^{\top} e_c$$"| Pred
  Pred --> MSE
  Compound --> Kernel
  Kernel --> RidgePred
  Strain -->|"$$\text{row } s$$"| RidgePred
  Pred --> Stack
  RidgePred --> Stack

  classDef input fill:#E1D5E7,stroke:#846592,color:#1a1a1a
  classDef embed fill:#FFE6CC,stroke:#BD8800,color:#1a1a1a
  classDef trans fill:#F8CECC,stroke:#A24A46,color:#1a1a1a
  classDef reg fill:#FFF2CC,stroke:#BCA04C,color:#1a1a1a
  classDef off fill:#E6E6E6,stroke:#666666,color:#1a1a1a
  class Strain,Compound,CellGraph input
  class Table,MLP,Seq,Hdel,Henv,Kernel embed
  class Transformer,DeleteOp,TokenLayer,Concat,Pred,RidgePred trans
  class MSE,Stack reg
  class GraphReg,Discard,Pooled off
```

What the diagram makes plain: the knockout enters the prediction through two 180-vectors,
both read at the deleted genes' own rows ($h^{\mathrm{del}}_s$ from the post-deletion
field, $h^{\mathrm{env}}_{sc}$ from the wildtype field after the compound token). Nothing
downstream of the deletion in the rest of the cell reaches the readout, and with the prior
off nothing about the wiring does either. The ridge half has no gene model at all. A model
that read fitness from the whole post-deletion, post-compound field (run the compound layer
over $H_{\mathrm{pert}}[s]$ per strain and pool, prior on) has not been run; that it would
help is a hypothesis.
