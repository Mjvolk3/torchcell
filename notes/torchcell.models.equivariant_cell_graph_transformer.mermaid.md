---
id: 37bi7cola1pitrigdqv0e34
title: Mermaid
desc: ''
updated: 1765214315720
created: 1765214315720
---

## Architecture Diagrams

This page contains mermaid architecture diagrams for the Cell Graph Transformer models.

**Main documentation:** [[torchcell.models.equivariant_cell_graph_transformer]]

## Non-Equivariant Cell Graph Transformer

The original architecture with immediate collapse to invariant outputs.

```mermaid
graph TD
  subgraph InputData["Input Data"]
    direction TB
    subgraph CellGraph["Cell Graph (Wildtype)"]
      direction TB
      cell_graph_genes["Genes: 6607 nodes"]
      cell_graph_edges["9 Graph Types:<br>physical, regulatory, tflink<br>STRING: neighborhood, fusion,<br>cooccurence, coexpression,<br>experimental, database"]
    end

    subgraph BatchData["Batch (Perturbations)"]
      direction TB
      batch_indices["perturbation_indices: [total_pert_genes]<br>perturbation_indices_batch: [total_pert_genes]<br>phenotype_values: [batch_size]"]
    end
  end

  subgraph ModelArch["Model Architecture"]
    direction TB

    subgraph Embeddings["Gene Embeddings + CLS"]
      direction TB
      gene_emb["Gene Embedding<br>nn.Embedding(6607, 96)<br>Init: Normal(0, 0.02)"]
      cls_token["CLS Token<br>nn.Parameter([1, 96])<br>Learnable"]
      concat_cls["Concatenate<br>[CLS || Gene Embeddings]<br>[1, N+1, 96]"]
    end

    subgraph TransformerStack["Transformer Encoder (8 Layers)"]
      direction TB

      subgraph Layer["GraphRegularizedTransformerLayer × 8"]
        direction TB

        subgraph Attention["Multi-Head Attention (12 heads)"]
          qkv_proj["Q, K, V Projections<br>Linear(96, 96) each"]
          attn_compute["Attention Computation<br>softmax(QK^T / √d_h)"]
          attn_output["Attention Output<br>[batch, heads, N+1, N+1]"]
        end

        subgraph GraphReg["Graph Regularization"]
          extract_gene_attn["Extract Gene-Gene Attn<br>[batch, heads, N, N]"]
          kl_div["KL Divergence<br>KL(A_tilde || α)"]
          reg_loss["Graph Reg Loss<br>Per head per graph"]
        end

        subgraph FFN["Feed Forward Network"]
          layernorm1["LayerNorm + Residual"]
          ffn_mlp["MLP: 96 → 384 → 96<br>GELU activation"]
          layernorm2["LayerNorm + Residual"]
        end
      end
    end

    subgraph PertHead["Perturbation Head (Fused Type I + II)"]
      direction TB

      subgraph HeadType["Switchable Mechanism"]
        direction LR
        cross_attn["Cross-Attention Path<br>MultiheadAttention(96, 8)"]
        hypersagnn["HyperSAGNN Path<br>Masked Self-Attention"]
      end

      subgraph Aggregation["Perturbation Aggregation"]
        extract_pert["Extract Perturbed Genes<br>H_genes[pert_indices]"]
        pool_or_attn["Mean Pool OR<br>Cross-Attend to All Genes"]
        combine_cls["Concatenate<br>[h_CLS || z_S]<br>[batch, 192]"]
      end

      subgraph PredMLP["Prediction MLP"]
        mlp_layers["MLP: 192 → 96 → 1<br>ReLU, Dropout"]
        predictions["Predictions<br>[batch_size, 1]"]
      end
    end
  end

  subgraph Outputs["Model Outputs"]
    direction TB
    pred_output["Predictions: [batch_size, 1]<br>INVARIANT"]

    subgraph Representations["Representations Dict"]
      h_cls_out["h_CLS: [96]"]
      h_genes_out["H_genes: [N, 96]"]
      attn_weights_out["attention_weights: List[Tensor]"]
      graph_reg_loss_out["graph_reg_loss: Scalar"]
    end
  end

  %% Main Flow Connections
  cell_graph_genes --> gene_emb
  gene_emb --> concat_cls
  cls_token --> concat_cls
  concat_cls --> Layer

  %% Transformer Layer Flow
  Layer --> qkv_proj
  qkv_proj --> attn_compute
  attn_compute --> attn_output
  attn_output --> extract_gene_attn

  %% Graph Regularization Flow
  cell_graph_edges --> kl_div
  extract_gene_attn --> kl_div
  kl_div --> reg_loss

  %% Attention to FFN
  attn_output --> layernorm1
  layernorm1 --> ffn_mlp
  ffn_mlp --> layernorm2

  %% Transformer to Perturbation Head
  layernorm2 --> extract_pert
  batch_indices --> extract_pert

  %% Perturbation Head Flow
  extract_pert --> cross_attn
  extract_pert --> hypersagnn
  cross_attn --> pool_or_attn
  hypersagnn --> pool_or_attn

  layernorm2 --> combine_cls
  pool_or_attn --> combine_cls
  combine_cls --> mlp_layers
  mlp_layers --> predictions

  %% Outputs
  predictions --> pred_output
  layernorm2 --> h_cls_out
  layernorm2 --> h_genes_out
  extract_gene_attn --> attn_weights_out
  reg_loss --> graph_reg_loss_out

  %% Styling
  classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px
  classDef input fill:#e1f5fe,stroke:#333,stroke-width:2px
  classDef embedding fill:#fff3e0,stroke:#333,stroke-width:2px
  classDef transformer fill:#e8f5e9,stroke:#333,stroke-width:2px
  classDef attention fill:#fce4ec,stroke:#333,stroke-width:2px
  classDef graph_reg fill:#f3e5f5,stroke:#333,stroke-width:2px
  classDef ffn fill:#e1f5fe,stroke:#333,stroke-width:2px
  classDef pert_head fill:#ffe8e8,stroke:#333,stroke-width:2px
  classDef output fill:#c8e6c9,stroke:#333,stroke-width:2px

  class InputData,CellGraph,BatchData input
  class Embeddings,gene_emb,cls_token,concat_cls embedding
  class TransformerStack,Layer transformer
  class Attention,qkv_proj,attn_compute,attn_output attention
  class GraphReg,extract_gene_attn,kl_div,reg_loss graph_reg
  class FFN,layernorm1,ffn_mlp,layernorm2 ffn
  class PertHead,HeadType,Aggregation,PredMLP,cross_attn,hypersagnn,extract_pert,pool_or_attn,combine_cls,mlp_layers,predictions pert_head
  class Outputs,Representations,pred_output,h_cls_out,h_genes_out,attn_weights_out,graph_reg_loss_out output
```

## Equivariant Cell Graph Transformer (Type I / Type II Separation)

The two-stage virtual instrument architecture preserving equivariance.

```mermaid
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

      GraphReg["Graph Regularization<br>KL(A_g || α^(ℓ,k))<br>for each graph type"]
    end
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

  %% Styling
  classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
  classDef embedding fill:#fff3e0,stroke:#f57c00,stroke-width:2px
  classDef transformer fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
  classDef typeI fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
  classDef typeII fill:#ffe0b2,stroke:#e65100,stroke-width:2px
  classDef equivariant fill:#ffebee,stroke:#c62828,stroke-width:2px,stroke-dasharray:5 5
  classDef invariant fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
  classDef sparse fill:#fff9c4,stroke:#f9a825,stroke-width:2px
  classDef output fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px

  class InputLayer,CellGraph,PerturbationData input
  class Embedding,GeneEmbed,CLSToken embedding
  class TransformerBlock,TransformerLayers,TLayer1,TLayer2,TLayerDots,TLayerL,GraphReg transformer
  class TypeIInstruments,PerturbationOps,DeleteOp,OverexpOp,KnockdownOp,CrossAttention,PerturbedState typeI
  class TypeIIInstruments typeII
  class InvariantReadouts,FitnessInst,GeneIntInst,MorphInst invariant
  class EquivariantReadouts,ExprInst,ProteinInst equivariant
  class SparseBatch,PtrArrays,MissingLabels,SparseLoss sparse
  class Outputs,OutFitness,OutGI,OutMorph,OutExpr output
```

## Simplified Data Flow Comparison

### Non-Equivariant (Immediate Collapse)

```mermaid
flowchart LR
    A["H_genes<br>[N, d]"] --> B["Extract S<br>[|S|, d]"]
    B --> C["Mean Pool<br>[d]"]
    C --> D["CrossAttn<br>[d]"]
    D --> E["[h_CLS || z_S]<br>[2d]"]
    E --> F["MLP<br>[1]"]
    F --> G["ŷ ∈ R<br>INVARIANT"]

    style G fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

### Equivariant (Type I / Type II)

```mermaid
flowchart LR
    subgraph TypeI["Type I Transform"]
        A["H_genes<br>[N, d]"] --> B["CrossAttn to S<br>[N, d]"]
        B --> C["LayerNorm<br>[N, d]"]
        C --> D["FFN<br>[N, d]"]
        D --> E["LayerNorm<br>[N, d]"]
    end

    E --> F["H_pert<br>[B, N, d]<br>EQUIVARIANT"]

    subgraph TypeII["Type II Readout"]
        F --> G["Pool S<br>[B, d]"]
        G --> H["[h_CLS || z_S]<br>[B, 2d]"]
        H --> I["MLP<br>[B, 1]"]
    end

    I --> J["ŷ ∈ R^B"]

    style F fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style TypeI fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px
    style TypeII fill:#ffe0b2,stroke:#e65100,stroke-width:1px
```

## Memory Comparison

```mermaid
graph TD
    subgraph WildtypeTransformer["Wildtype Transformer<br>(Shared by both models)"]
        WT_Input["Input: [1, N+1, d]"]
        WT_Attn["Attention: [1, H, N, N]<br>~1.4 GB/layer"]
        WT_Output["Output: [1, N+1, d]"]
    end

    subgraph NonEquivariant["Non-Equivariant Path"]
        NE_Collapse["Immediate Collapse<br>[B, d]"]
        NE_Memory["Memory: ~10-12 GB<br>Batch: 512 ✓"]
    end

    subgraph Equivariant["Equivariant Path"]
        EQ_Transform["Type I Transform<br>[B, N, d]"]
        EQ_Memory["Memory: ~17 GB<br>Batch: 256 ✓"]
    end

    subgraph Propagation["With Propagation Layers<br>(Memory Prohibitive)"]
        Prop_Attn["Attention: [B, H, N, N]<br>~357 GB at B=256"]
        Prop_Memory["Exceeds 4xA100 capacity<br>OOM at any batch size ❌"]
    end

    WT_Output --> NE_Collapse
    WT_Output --> EQ_Transform
    EQ_Transform -.->|"Not Implemented"| Prop_Attn

    style Propagation fill:#ffebee,stroke:#c62828,stroke-width:2px
    style NonEquivariant fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Equivariant fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

## Architecture Design Principles

```mermaid
mindmap
  root((Virtual Cell Transformer))
    id1(Type I)
      Perturbation Transform
      Equivariant Output
      B x N x d tensor
      Modular Design
    id2(Type II)
      Task Readouts
      Swappable Heads
      Fitness Prediction
      Gene Interaction
      Expression Levels
      Morphology Features
    id3(Graph Regularization)
      KL Divergence Loss
      Per head per graph
      Mid layer application
    id4(Memory Efficiency)
      Shared wildtype encoder
      No propagation layers
      Cross attention only
```
