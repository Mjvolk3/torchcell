---
id: k4nj6777uuufvlkww6poyn8
title: Output Side
desc: ''
updated: 1792449139586
created: 1792449139586
---

## 2026.10.19 - The output side of the 043 model

How a readout leaves the perturbed graph, by readout type, with the experiment-type,
dataset and environment tokens; the per-dataset label tensors and their masks; the losses;
and the three evaluation splits plus the morphology probe. Prose and the facts behind every
choice: `notes-tex/multimodal/043-datasets-primer/sections/output-design.tex`
([[experiments.043-expression-proteome-morphology]]). Prototype of the heads and the loss:
`torchcell/models/readout_heads_043.py`.

Palette and rules as in
[[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]]: one KaTeX block
per label, `\\\` row breaks, four primary colors. Purple: the perturbed graph and the
tokens (inputs to the heads). Orange: the heads. Red: label tensors and losses. Yellow:
the evaluation.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph Sites["$$\text{Readout sites: the perturbed graph}$$"]
    direction LR
    CLS["$$\begin{gathered}\text{Perturbed CLS}\\\ h \in \mathbb{R}^{d}\end{gathered}$$"]
    ZS["$$\begin{gathered}\text{Perturbed-gene summary}\\\ z_S = \sum_{g \in S} H_g\end{gathered}$$"]
    HG["$$\begin{gathered}\text{Gene states}\\\ H \in \mathbb{R}^{N \times d},\ N = 6607\\\ \text{gene } g \text{ reads } H_g \text{ only}\end{gathered}$$"]
  end

  subgraph Tokens["$$\text{Tokens}\ t = [\,t_{\mathrm{type}} ; t_{\mathrm{data}} ; t_{\mathrm{env}}\,]$$"]
    direction LR
    TType["$$\begin{gathered}\text{Experiment type}\\\ \text{ontology phenotype class}\\\ \text{flat id or DAG path (owner)}\end{gathered}$$"]
    TData["$$\begin{gathered}\text{Dataset}\\\ 15\ \text{ids}\end{gathered}$$"]
    TEnv["$$\begin{gathered}\text{Environment}\\\ \text{medium, temperature, treatment}\end{gathered}$$"]
  end

  subgraph Heads["$$\text{Heads by readout type, joined; per-dataset affine } a_k, b_k$$"]
    direction LR
    HExpr["$$\begin{gathered}\text{Expression log ratio}\\\ \hat y_g = a_k f([H_g ; t]) + b_k\\\ \text{Kemmeren, Sameith Dm, Nadal}\end{gathered}$$"]
    HAbs["$$\begin{gathered}\text{Absolute expression}\\\ \log_2(\mathrm{CPM}{+}1),\ \log_2(\mathrm{TPM}{+}1)\\\ \text{Jackson, Caudal, Boocock, N'Guessan}\end{gathered}$$"]
    HScore["$$\begin{gathered}\text{Expression score}\\\ \text{Hu 2007 X score}\end{gathered}$$"]
    HProt["$$\begin{gathered}\text{Protein abundance}\\\ \text{Messner, Zelezniak, Muenzner}\end{gathered}$$"]
    HTurn["$$\begin{gathered}\text{Protein turnover}\\\ \text{Muenzner}\end{gathered}$$"]
    HProm["$$\begin{gathered}\text{Promoter level}\\\ \hat y = \mathrm{MLP}([h ; z_S ; t])\\\ \text{Vaishnav}\end{gathered}$$"]
  end

  subgraph Labels["$$\text{Per-dataset label tensors, measured mask}\ m_{b,g} = \neg\,\mathrm{isnan}(y_{b,g})$$"]
    direction LR
    LDel["$$\begin{gathered}\text{Deletion panels}\\\ Y \in \mathbb{R}^{B \times N},\ \text{NaN unmeasured}\\\ \text{Kemmeren } 1482 \times 6169\end{gathered}$$"]
    LNat["$$\begin{gathered}\text{Isolates, segregants}\\\ \text{centered per gene over the panel}\\\ \text{Caudal } 943,\ \text{Muenzner } 795\end{gathered}$$"]
    LScal["$$\begin{gathered}\text{Scalar}\\\ y \in \mathbb{R}^{B}\\\ \text{Vaishnav } 71158\end{gathered}$$"]
  end

  subgraph Loss["$$\text{Losses, measured entries only}$$"]
    direction LR
    LMSE["$$\begin{gathered}\text{MSE (base)}\\\ \tfrac{1}{\sum m}\sum_{b,g} m_{b,g}(\hat y_{b,g} - y_{b,g})^2\end{gathered}$$"]
    LPin["$$\begin{gathered}\text{19-quantile pinball (arm)}\\\ \tfrac{1}{K}\sum_k \rho_{\tau_k}(y - \hat q_{\tau_k})\end{gathered}$$"]
    LSum["$$\begin{gathered}\mathcal{L} = \sum_{\text{type}} w_{\text{type}}\, \ell_{\text{type}}\\\ \text{equal weight per dataset (base)}\\\ \text{record count (arm)}\end{gathered}$$"]
  end

  subgraph Eval["$$\text{Evaluation, declared before any read}$$"]
    direction LR
    SA["$$\begin{gathered}\text{Split A: Kemmeren by gene}\\\ 148\ \text{held-out deletions}\\\ \text{per-gene } r \text{ across strains}\\\ \text{ceiling } 0.729\end{gathered}$$"]
    SB["$$\begin{gathered}\text{Split B: all 72 Sameith doubles}\\\ \text{vs measured, vs } M_x + M_y\\\ \text{vs stronger single}\end{gathered}$$"]
    SC["$$\begin{gathered}\text{Split C: Muenzner isolates}\\\ 159\ \text{held out of proteome,}\\\ \text{Caudal and turnover}\\\ \text{reference } 0.208\end{gathered}$$"]
    Probe["$$\begin{gathered}\text{Morphology probe: Ohya 2005}\\\ \text{ridge from frozen } h,\ z_S,\ \bar H\\\ 116\ \text{moving features, 5-fold}\\\ \text{bars } 0.149,\ 0.348,\ 0.847\end{gathered}$$"]
  end

  %% Stack the tokens under the sites (both are sources) so the figure stays narrow.
  Sites ~~~ Tokens

  HG --> HExpr
  HG --> HAbs
  HG --> HScore
  HG --> HProt
  HG --> HTurn
  CLS --> HProm
  ZS --> HProm
  Tokens --> Heads

  HExpr --> LDel
  HScore --> LDel
  HProt --> LDel
  HAbs --> LNat
  HProt --> LNat
  HTurn --> LNat
  HAbs --> LDel
  HProm --> LScal

  Labels --> LMSE
  Labels --> LPin
  LMSE --> LSum
  LPin --> LSum

  LSum --> SA
  LSum --> SB
  LSum --> SC
  Sites -.->|"$$\text{frozen, no training on Ohya}$$"| Probe

  classDef sites fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef tokens fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef heads fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef labels fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef loss fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef eval fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px

  class CLS,ZS,HG sites
  class TType,TData,TEnv tokens
  class HExpr,HAbs,HScore,HProt,HTurn,HProm heads
  class LDel,LNat,LScal labels
  class LMSE,LPin,LSum loss
  class SA,SB,SC,Probe eval
```

Render:

```bash
bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.043-expression-proteome-morphology.mermaid.output-side.md
# -> notes/assets/pdf-output/experiments.043-expression-proteome-morphology.mermaid.output-side.pdf
```

Numbers in the boxes: Kemmeren 1,482 genotypes and 6,169 genes, Caudal 943, Muenzner 795,
Vaishnav 71,158 (store table, [[experiments.043-expression-proteome-morphology]]); the
Kemmeren ceiling 0.729 (within-dataset ceilings, job 5893); the Caudal x Muenzner
per-isolate reference 0.208 (cross-dataset correlation, job 5892); the probe bars 0.149
(019 genotype-only morphology, M_zs with MSE, 12 runs), 0.348 (Kemmeren ridge, 1,437
genotypes) and 0.847 (019 ceiling on the 116 moving features).
