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
  subgraph InputLayer["Input Layer"]
    direction TB
    CellGraph["Cell Graph<br>|G| = 6607 genes<br>9 graph types"]
    PerturbationData["Perturbation Data<br>S ⊆ G, τ ∈ (del, OE, KD)"]
  end

  subgraph TransformerBlock["Graph-Regularized Transformer"]
    direction TB

    subgraph Embedding["Embeddings"]
      GeneEmbed["Gene Embeddings<br>H^(0) ∈ R^(N+1)×d"]
      CLSToken["[CLS] Token"]
    end

    subgraph TransformerLayers["L Transformer Layers"]
      direction TB
      TLayer1["Layer 1: H^(1) = T^(1)(H^(0))"]
      TLayer2["Layer 2: H^(2) = T^(2)(H^(1))"]
      TLayerDots["..."]
      TLayerL["Layer L: H^(L) = T^(L)(H^(L-1))"]
    end

    GraphReg["Graph Regularization<br>KL(A_g || α^(ℓ,k))<br>for each graph type"]
  end

  subgraph TypeIInstruments["Type I Virtual Instruments<br>(Representation → Representation)"]
    direction TB

    subgraph PerturbationOps["Perturbation Operators"]
      DeleteOp["T_ψ^del: Gene Deletion"]
      OverexpOp["T_ψ^OE: Overexpression"]
      KnockdownOp["T_ψ^KD: Knockdown"]
    end

    CrossAttention["Cross-Attention Mechanism<br>H_pert,i = h_i + Σ_j∈S α_ij W_V h_j"]

    PerturbedState["H_pert ∈ R^(B×N×d)<br>EQUIVARIANT"]
  end

  subgraph TypeIIInstruments["Type II Virtual Instruments<br>(Representation → Output)"]
    direction TB

    subgraph InvariantReadouts["Invariant Readouts"]
      FitnessInst["R_φ^fit: Fitness<br>GlobalPool → MLP → R"]
      GeneIntInst["R_φ^GI: Gene Interaction<br>Pool_S → MLP → R"]
      MorphInst["R_φ^morph: Morphology<br>GeneSetPool → MLP → R^m"]
    end

    subgraph EquivariantReadouts["Equivariant Readouts"]
      ExprInst["R_φ^expr: Expression<br>Per-gene MLP → R^N"]
      ProteinInst["R_φ^prot: Protein Levels<br>Per-gene MLP → R^N"]
    end
  end

  subgraph SparseBatch["Sparse Batch Handler"]
    direction TB
    PtrArrays["Pointer Arrays<br>ptr_t for each task"]
    MissingLabels["Missing Label Masks<br>n_b^(t) ∈ {0,1}"]
    SparseLoss["Sparse Loss Computation<br>L = Σ_t w_t Σ_b ℓ_t"]
  end

  subgraph Outputs["Multi-Task Outputs"]
    direction TB
    OutFitness["y_fitness ∈ R"]
    OutGI["y_GI ∈ R"]
    OutMorph["y_morph ∈ R^m"]
    OutExpr["y_expr ∈ R^N"]
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
  GraphReg -.->|regularize| TransformerLayers

  TLayerL --> PerturbationOps
  PerturbationOps --> CrossAttention
  CrossAttention --> PerturbedState

  PerturbedState --> FitnessInst
  PerturbedState --> GeneIntInst
  PerturbedState --> MorphInst
  PerturbedState --> ExprInst
  PerturbedState --> ProteinInst

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
