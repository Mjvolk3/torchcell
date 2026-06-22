---
id: 3x38y7uako89u0txjxr8gk3
title: Type I Ii
desc: ''
updated: 1781756443562
created: 1781756443562
---

## 2026.06.17 - Type I / Type II diagram (new palette)

The Equivariant Cell Graph Transformer (Type I / Type II separation) diagram from
[[torchcell.models.equivariant_cell_graph_transformer.mermaid]] (L170-302), recolored with
the draw.io-aligned palette defined in
[[torchcell.models.equivariant_cell_graph_transformer.mermaid.colors]]. Beige background via
`%%{init ...}%%`; large container subgraphs are left neutral (color sits on the smaller boxes).

Class -> color mapping. **Base primary only** -- four colors (orange / red / purple / yellow), solid fills, no secondary tier, no outlines, no dashes. Colors group related stages:

| Color | Classes | Fill / border |
|---|---|---|
| Purple | input, equivariant readouts | `#E1D5E7` / `#846592` |
| Orange | embedding, Type I virtual instruments, sparse batch | `#FFE6CC` / `#BD8800` |
| Red | transformer layers, output | `#F8CECC` / `#A24A46` |
| Yellow | invariant readouts, graph regularization | `#FFF2CC` / `#BCA04C` |

Large outer container subgraphs are left neutral beige; alt colors (blue/green/grey) and the secondary tier are unused here.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph InputLayer["$$\text{Input Layer}$$"]
    direction TB
    CellGraph["$$\begin{gathered}\text{Cell Graph}\\\ |G| = 6607\ \text{genes}\\\ \text{9 graph types}\end{gathered}$$"]
    PerturbationData["$$\begin{gathered}\text{Perturbation Data}\\\ S \subseteq G,\ \tau \in \{\mathrm{del}, \mathrm{OE}, \mathrm{KD}\}\end{gathered}$$"]
  end

  subgraph TransformerBlock["$$\text{Graph-Regularized Transformer}$$"]
    direction TB

    subgraph Embedding["$$\text{Embeddings}$$"]
      GeneEmbed["$$\begin{gathered}\text{Gene Embeddings}\\\ H^{(0)} \in \mathbb{R}^{(N+1)\times d}\end{gathered}$$"]
      CLSToken["$$[\text{CLS}]\ \text{Token}$$"]
    end

    subgraph TransformerLayers["$$L\ \text{Transformer Layers}$$"]
      direction TB
      TLayer1["$$\begin{gathered}\text{Layer 1}\\\ H^{(1)} = T^{(1)}(H^{(0)})\end{gathered}$$"]
      TLayer2["$$\begin{gathered}\text{Layer 2}\\\ H^{(2)} = T^{(2)}(H^{(1)})\end{gathered}$$"]
      TLayerDots["$$\cdots$$"]
      TLayerL["$$\begin{gathered}\text{Layer L}\\\ H^{(L)} = T^{(L)}(H^{(L-1)})\end{gathered}$$"]
    end

    GraphReg["$$\begin{gathered}\text{Graph Regularization}\\\ \mathrm{KL}(A_g \,\|\, \alpha^{(\ell,k)})\\\ \text{for each graph type}\end{gathered}$$"]
  end

  subgraph TypeIInstruments["$$\begin{gathered}\text{Type I Virtual Instruments}\\\ (\text{Representation} \to \text{Representation})\end{gathered}$$"]
    direction TB

    subgraph PerturbationOps["$$\text{Perturbation Operators}$$"]
      DeleteOp["$$\begin{gathered}T_\psi^{\mathrm{del}}\\\ \text{Gene Deletion}\end{gathered}$$"]
      OverexpOp["$$\begin{gathered}T_\psi^{\mathrm{OE}}\\\ \text{Overexpression}\end{gathered}$$"]
      KnockdownOp["$$\begin{gathered}T_\psi^{\mathrm{KD}}\\\ \text{Knockdown}\end{gathered}$$"]
    end

    CrossAttention["$$\begin{gathered}\text{Cross-Attention Mechanism}\\\ H_{\mathrm{pert},i} = h_i + \sum_{j \in S} \alpha_{ij} W_V h_j\end{gathered}$$"]

    PerturbedState["$$\begin{gathered}H_{\mathrm{pert}} \in \mathbb{R}^{B \times N \times d}\\\ \text{EQUIVARIANT}\end{gathered}$$"]
  end

  subgraph TypeIIInstruments["$$\begin{gathered}\text{Type II Virtual Instruments}\\\ (\text{Representation} \to \text{Output})\end{gathered}$$"]
    direction TB

    subgraph InvariantReadouts["$$\text{Invariant Readouts}$$"]
      FitnessInst["$$\begin{gathered}R_\phi^{\mathrm{fit}}\text{: Fitness}\\\ \mathrm{GlobalPool} \to \mathrm{MLP} \to \mathbb{R}\end{gathered}$$"]
      GeneIntInst["$$\begin{gathered}R_\phi^{\mathrm{GI}}\text{: Gene Interaction}\\\ \mathrm{Pool}_S \to \mathrm{MLP} \to \mathbb{R}\end{gathered}$$"]
      MorphInst["$$\begin{gathered}R_\phi^{\mathrm{morph}}\text{: Morphology}\\\ \mathrm{GeneSetPool} \to \mathrm{MLP} \to \mathbb{R}^m\end{gathered}$$"]
    end

    subgraph EquivariantReadouts["$$\text{Equivariant Readouts}$$"]
      ExprInst["$$\begin{gathered}R_\phi^{\mathrm{expr}}\text{: Expression}\\\ \text{Per-gene}\ \mathrm{MLP} \to \mathbb{R}^N\end{gathered}$$"]
      ProteinInst["$$\begin{gathered}R_\phi^{\mathrm{prot}}\text{: Protein Levels}\\\ \text{Per-gene}\ \mathrm{MLP} \to \mathbb{R}^N\end{gathered}$$"]
    end
  end

  subgraph SparseBatch["$$\text{Sparse Batch Handler}$$"]
    direction TB
    PtrArrays["$$\begin{gathered}\text{Pointer Arrays}\\\ \mathrm{ptr}_t\ \text{for each task}\end{gathered}$$"]
    MissingLabels["$$\begin{gathered}\text{Missing Label Masks}\\\ n_b^{(t)} \in \{0,1\}\end{gathered}$$"]
    SparseLoss["$$\begin{gathered}\text{Sparse Loss Computation}\\\ \mathcal{L} = \sum_t w_t \sum_b \ell_t\end{gathered}$$"]
  end

  subgraph Outputs["$$\text{Multi-Task Outputs}$$"]
    direction TB
    OutFitness["$$y_{\mathrm{fitness}} \in \mathbb{R}$$"]
    OutGI["$$y_{\mathrm{GI}} \in \mathbb{R}$$"]
    OutMorph["$$y_{\mathrm{morph}} \in \mathbb{R}^m$$"]
    OutExpr["$$y_{\mathrm{expr}} \in \mathbb{R}^N$$"]
  end

  %% Main Flow
  CellGraph --> GeneEmbed
  PerturbationData --> PerturbationOps

  GeneEmbed --> TLayer1
  CLSToken --> TLayer1
  TLayer1 --> TLayer2
  TLayer2 --> TLayerDots
  TLayerDots --> TLayerL

  CellGraph --> GraphReg
  GraphReg -.->|"$$\text{regularize}$$"| TransformerLayers

  TLayerL --> PerturbationOps
  PerturbationOps --> CrossAttention
  CrossAttention --> PerturbedState

  PerturbedState --> FitnessInst
  PerturbedState --> GeneIntInst
  PerturbedState --> MorphInst
  PerturbedState --> ExprInst
  PerturbedState --> ProteinInst

  %% Stack Type II readouts into two rows (Equivariant below Invariant) to reduce width
  InvariantReadouts ~~~ EquivariantReadouts

  FitnessInst --> OutFitness
  GeneIntInst --> OutGI
  MorphInst --> OutMorph
  ExprInst --> OutExpr

  OutFitness --> SparseLoss
  OutGI --> SparseLoss
  OutMorph --> SparseLoss
  OutExpr --> SparseLoss

  MissingLabels --> SparseLoss
  PtrArrays --> SparseLoss

  %% Styling: base primary only (orange/red/purple/yellow); solid fills, no secondary/outline/dashes.
  classDef input fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef embedding fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef transformer fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef graphreg fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef typeI fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef invariant fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef equivariant fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef sparse fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef output fill:#F8CECC,stroke:#A24A46,stroke-width:2px

  class CellGraph,PerturbationData input
  class Embedding,GeneEmbed,CLSToken embedding
  class TransformerLayers,TLayer1,TLayer2,TLayerDots,TLayerL transformer
  class GraphReg graphreg
  class PerturbationOps,DeleteOp,OverexpOp,KnockdownOp,CrossAttention,PerturbedState typeI
  class InvariantReadouts,FitnessInst,GeneIntInst,MorphInst invariant
  class EquivariantReadouts,ExprInst,ProteinInst equivariant
  class PtrArrays,MissingLabels,SparseLoss sparse
  class OutFitness,OutGI,OutMorph,OutExpr output
  %% large outer containers (InputLayer, TransformerBlock, TypeI/II Instruments, SparseBatch, Outputs) left unstyled -> neutral beige
```

## 2026.06.21 - All-LaTeX labels + render to PDF

Every label, cluster title, and edge label was converted to a single KaTeX block
(no mixed text/math on one line; multi-line via `\begin{gathered}...\\...\end{gathered}`)
so math renders cleanly with consistent spacing. See the pipeline + authoring rule
in [[paper.nature-biotech.figures]].

Regenerate the figure asset (name matches this note):

```bash
bash notes/assets/publish/scripts/mermaid_pdf.sh notes/torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii.md
# -> notes/assets/pdf-output/torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii.pdf
```
