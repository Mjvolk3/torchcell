---
id: 6e7wbr3sg6q0wiggiymoufu
title: Readout Concat
desc: ''
updated: 1789285003867
created: 1789285003867
---

## 2026.09.13 - What the per-gene readout sees: H_ref against H_concat

The v12 head round's winning arm, `H_concat` (`concat_context: true`), changes only the
input of the per-gene readout MLP. Everything above it (encoder, perturbation
cross-attention, residual, FFN) is identical to `H_ref`. The diagram follows one strain
$b$ with deletion set $S_b$ and one gene $i$ through the Type I instrument to the
expression prediction $\hat y_{b,i}$. Same palette as
[[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]].

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph Encoder["$$\text{Graph-regularized transformer, shared by all strains}$$"]
    direction TB
    Hgenes["$$\begin{gathered}\text{gene tokens}\ H \in \mathbb{R}^{N \times d}\\\ h_i = \text{row}\ i,\ \text{strain-invariant}\end{gathered}$$"]
  end

  subgraph PertData["$$\text{Strain}\ b$$"]
    Sb["$$\begin{gathered}\text{deletion set}\ S_b \subseteq G\\\ \text{e.g.}\ \{\mathit{rpd3}\}\ \text{or}\ \{\mathit{fkh1}, \mathit{yap6}\}\end{gathered}$$"]
  end

  subgraph TypeI["$$\text{Type I instrument: EquivariantPerturbationTransform}$$"]
    direction TB
    KV["$$\begin{gathered}\text{keys, values} = H[S_b]\\\ \text{the tokens of the deleted genes}\end{gathered}$$"]
    Attn["$$\begin{gathered}\text{cross-attention, query} = h_i\\\ c_{b,i} = \mathrm{Attn}(h_i;\ H[S_b])\\\ \text{one}\ d\text{-vector per gene per strain}\end{gathered}$$"]
    Resid["$$\begin{gathered}\text{residual + norm + FFN}\\\ h^{\mathrm{pert}}_{b,i} = \mathrm{FFN}\big(\mathrm{LN}(h_i + c_{b,i})\big)\\\ \text{sees only the SUM}\end{gathered}$$"]
  end

  subgraph Href["$$\text{H\_ref readout}$$"]
    RefIn["$$\text{input} = [\,h^{\mathrm{pert}}_{b,i}\,] \in \mathbb{R}^{d}$$"]
    RefMLP["$$\begin{gathered}\text{shared 2-layer MLP}\\\ \hat y_{b,i} = f(h^{\mathrm{pert}}_{b,i})\end{gathered}$$"]
  end

  subgraph Hconcat["$$\text{H\_concat readout (State SE form)}$$"]
    CatIn["$$\begin{gathered}\text{input} = [\,h^{\mathrm{pert}}_{b,i}\ ;\ h_i\ ;\ c_{b,i}\,] \in \mathbb{R}^{3d}\\\ \text{perturbed token ; gene identity ; strain context}\end{gathered}$$"]
    CatMLP["$$\begin{gathered}\text{shared 2-layer MLP, width}\ 3d\\\ \hat y_{b,i} = f(h^{\mathrm{pert}}_{b,i}, h_i, c_{b,i})\end{gathered}$$"]
  end

  Out["$$\begin{gathered}\hat y_{b,i}\ \text{for every gene}\ i\\\ \text{19 pinball knots, median scored}\end{gathered}$$"]

  Hgenes --> Attn
  Sb --> KV
  Hgenes --> KV
  KV --> Attn
  Attn --> Resid
  Resid --> RefIn
  RefIn --> RefMLP
  RefMLP --> Out
  Resid --> CatIn
  Hgenes -. "$$h_i$$" .-> CatIn
  Attn -. "$$c_{b,i}\ \text{before the residual}$$" .-> CatIn
  CatIn --> CatMLP
  CatMLP --> Out

  classDef input fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef embedding fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef transformer fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef typeI fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef equivariant fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef output fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  class Sb input
  class Hgenes transformer
  class KV,Attn,Resid typeI
  class RefIn,RefMLP,CatIn,CatMLP equivariant
  class Out output
```

Reading. $h_i$ is the gene's embedding after the shared encoder; it does not depend on
the strain. $c_{b,i}$ is what gene $i$ reads off the deleted genes of strain $b$ by
cross-attention; it depends on both. `H_ref` folds the two into one token
$h^{\mathrm{pert}} = \mathrm{FFN}(\mathrm{LN}(h_i + c_{b,i}))$ and the readout sees only that,
so the prediction is a function of the sum, and the layer norm has discarded the
magnitude of the sum, where $\langle h_i, c_{b,i} \rangle$ lives. `H_concat` hands the
readout the same token plus the two unnormalized parts side by side, so it can learn
any interaction between "which gene" and "what was deleted". Measured (v12, four
seeds): $+0.0135$ over `H_ref`, positive in every seed.
[[experiments.019-simb-multimodal.scripts.head_round_readout]]
