---
id: k90s9jp976ob7sz2na63s65
title: Hill Bliss Model
desc: ''
updated: 1792379523412
created: 1792379523412
---

## 2026.10.18 - The leading round-11 arm, wired

The arm `van_e150_cell_bliss_hill_anchors` of round 11 (`conf/mixture/r11_van_e150_cell_bliss_hill_anchors.yaml`, trainer `experiments/040-inhibitor-synergy-wetlab/scripts/train_mixture.py`): the cell readout, Bliss split, a Hill curve per inhibitor on the host, trained on Vanacloig 2022 and the public IC30 anchors only. Every size and setting below is read from the trainer and that config; scores live in the gate document's generated tables, not here. With the graph-prior weight at zero the nine gene graphs enter only through a KL penalty that is computed and not added, so no graph signal reaches this arm. Rendered with `bash notes/assets/publish/scripts/mermaid_pdf.sh` to `notes/assets/pdf-output/experiments.040-inhibitor-synergy-wetlab.mermaid.hill-bliss-model.pdf`, which the gate document places as SI figure 8.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph Inputs["$$\text{Inputs}$$"]
    direction LR
    Genes["$$\begin{gathered}\text{Gene tokens}\\\ 6607\ \text{genes, learned table}\\\ d = 180\end{gathered}$$"]
    Genotype["$$\begin{gathered}\text{Genotype}\\\ \text{deleted set}\ S\\\ S = \varnothing\ \text{for the host}\end{gathered}$$"]
    Compound["$$\begin{gathered}\text{Compound}\\\ \text{FCFP4 counts, 2048 bits}\\\ \text{standardized on fitted compounds}\end{gathered}$$"]
    Dose["$$\begin{gathered}\text{Dose}\\\ z = (\log_{10} M - \mu)/\sigma\\\ \text{over the fitted doses}\end{gathered}$$"]
  end

  subgraph Encoder["$$\text{Cell graph transformer, one layer}$$"]
    direction TB
    Pre["$$\begin{gathered}\text{Preprocessor}\\\ \text{2-layer MLP, dropout 0.1}\end{gathered}$$"]
    TL["$$\begin{gathered}\text{Transformer layer}\\\ 9\ \text{heads over all genes}\\\ \text{graph KL weight}\ \lambda = 0\end{gathered}$$"]
    H["$$\begin{gathered}\text{Wild-type gene states}\\\ H \in \mathbb{R}^{6607 \times 180}\end{gathered}$$"]
    Del["$$\begin{gathered}\text{Deletion operator}\\\ \text{every gene cross-attends to}\ S\\\ H^{\mathrm{pert}} = T(H, S)\end{gathered}$$"]
  end

  subgraph Medium["$$\text{The medium perturbs the cell}$$"]
    direction TB
    CMLP["$$\begin{gathered}\text{Compound MLP}\\\ 2048 \to 256 \to 180\\\ \text{GELU, dropout 0.2}\end{gathered}$$"]
    Split["$$\begin{gathered}\text{Bliss split}\\\ k\text{-compound medium} \to k\ \text{single media}\end{gathered}$$"]
    ML["$$\begin{gathered}\text{Medium layer}\\\ [\,e\,;\,H\,]\ \text{pre-norm attention, 9 heads}\end{gathered}$$"]
    Hc["$$\begin{gathered}\text{Cell in the medium}\\\ H^{c},\ \text{compound token after reading}\ e'\end{gathered}$$"]
  end

  subgraph Readout["$$\text{One readout for every record}$$"]
    direction TB
    State["$$\begin{gathered}\text{Cell state}\\\ s = [\,H^{c}_{S}\,;\,H^{\mathrm{pert}}_{S}\,;\,\overline{H^{c}}\,;\,e'\,;\,t_{\mathrm{source}}\,] \in \mathbb{R}^{900}\\\ \text{LayerNorm; zero}\ S\ \text{blocks for the host}\end{gathered}$$"]
    GHead["$$\begin{gathered}\text{Gene head}\\\ \mathrm{MLP}\ 900 \to 256 \to 1\\\ +\ \text{offset}(e) + \text{gene bias}\end{gathered}$$"]
    Hill["$$\begin{gathered}\text{Hill head, host}\\\ \text{linear}\ 900 \to 2:\ z_{50},\ s\\\ n = \mathrm{softplus}(s + s_0)\end{gathered}$$"]
    Curve["$$\begin{gathered}\text{Single-agent fitness}\\\ f_i = \sigma\!\left(-n\,(z_i - z_{50})\right)\\\ s_0:\ \text{Hill coefficient 1 in molar}\end{gathered}$$"]
    Bliss["$$\begin{gathered}\text{Bliss composition}\\\ F = \prod_{i=1}^{k} f_i\end{gathered}$$"]
  end

  subgraph Train["$$\text{Training}$$"]
    direction TB
    Loss["$$\begin{gathered}\text{Loss}\\\ \mathcal{L} = \mathrm{MSE}_{\mathrm{Vanacloig}} + \lambda_{\mathrm{host}}\,\mathrm{MSE}_{\mathrm{host}},\ \ \lambda_{\mathrm{host}} = 1\end{gathered}$$"]
    Anchors["$$\begin{gathered}\text{Host records, zero-shot}\\\ \text{public IC30 dose} \to F = 0.70\\\ \text{no compound} \to F = 1\\\ \text{no titration or combination well}\end{gathered}$$"]
    Opt["$$\begin{gathered}\text{AdamW, lr}\ 10^{-3},\ \text{encoder}\ 5 \times 10^{-4}\\\ \text{weight decay}\ 10^{-4},\ \text{OneCycle, 150 epochs}\\\ \text{gene batch 128, host batch 16}\\\ \text{select on 4 validation compounds}\end{gathered}$$"]
  end

  subgraph Eval["$$\text{Scored on}$$"]
    direction LR
    Pub["$$\begin{gathered}\text{Vanacloig}\\\ \text{held-out compounds,}\\\ \text{5 folds, 3 seeds}\end{gathered}$$"]
    Wet["$$\begin{gathered}\text{Bioscreen host}\\\ \text{titrations, 63 combinations,}\\\ \text{3 isobole grids}\end{gathered}$$"]
  end

  Genes --> Pre --> TL --> H
  H --> Del
  Genotype --> Del
  Compound --> CMLP --> Split --> ML
  H --> ML --> Hc
  Hc --> State
  Del --> State
  State --> GHead --> Loss
  State --> Hill --> Curve
  Dose --> Curve
  Curve --> Bliss --> Loss
  Anchors --> Loss
  Loss --> Opt
  GHead --> Pub
  Bliss --> Wet

  classDef input fill:#E1D5E7,stroke:#846592,color:#000
  classDef op fill:#FFE6CC,stroke:#BD8800,color:#000
  classDef layer fill:#F8CECC,stroke:#A24A46,color:#000
  classDef readout fill:#FFF2CC,stroke:#BCA04C,color:#000
  class Genes,Genotype,Compound,Dose,Anchors input
  class Pre,Del,CMLP,Split,Curve,Bliss op
  class TL,ML,H,Hc,Loss,Opt layer
  class State,GHead,Hill,Pub,Wet readout
```
