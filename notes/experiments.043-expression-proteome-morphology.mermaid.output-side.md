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
  subgraph Sites["$$\text{Readout sites of the perturbed graph}$$"]
    direction LR
    HG["$$\begin{gathered}\text{gene states}\ H_g\\\ N = 6{,}607\end{gathered}$$"]
    CZ["$$\begin{gathered}\text{CLS}\ h\ \text{and summary}\\\ z_S = \textstyle\sum_{g \in S} H_g\end{gathered}$$"]
    Tok["$$\begin{gathered}\text{tokens at the head}\\\ t_{\mathrm{type}}\ \text{(ontology path)},\ t_{\mathrm{env}},\ t_{\mathrm{data}}\end{gathered}$$"]
  end

  subgraph Heads["$$\text{One head per readout level}$$"]
    direction LR
    HGene["$$\begin{gathered}\text{gene-level head}\\\ \hat z_g = f([H_g ; t]) + c_d\\\ \text{11 datasets, every type}\end{gathered}$$"]
    HRec["$$\begin{gathered}\text{record-level head}\\\ \hat z = f([h ; z_S ; t]) + c_d\\\ \text{Vaishnav}\end{gathered}$$"]
  end

  subgraph Loss["$$\text{Loss on standardized targets, measured entries}$$"]
    direction LR
    Std["$$\begin{gathered}z = (y - \mu_d)/s_d\\\ \text{training rows only}\\\ \text{NaN masked}\end{gathered}$$"]
    Bins["$$\begin{gathered}\text{HL-Gauss, main}\\\ K = 32\ \text{over}\ [-4, 4]\\\ \sigma = 0.75\ \text{bins}\end{gathered}$$"]
    Arms["$$\begin{gathered}\text{arms: pinball}\\\ \text{one MSE cell}\\\ \text{equal weight per dataset}\end{gathered}$$"]
  end

  subgraph Eval["$$\text{Evaluation, declared before any read}$$"]
    direction LR
    Lin["$$\begin{gathered}\text{linear ladder first}\\\ \text{CPU, frozen tokens}\\\ \text{N0, ridge, kNN, GBLUP}\end{gathered}$$"]
    Splits["$$\begin{gathered}\text{fold file splits A to E}\\\ \text{gain}\ \geq 0.02,\ 5\ \text{of}\ 6\ \text{seeds}\\\ \text{interval over strains}\end{gathered}$$"]
    Probe["$$\begin{gathered}\text{Ohya probe, frozen}\\\ \text{raw and growth-residualized}\\\ \text{bars}\ 0.149,\ 0.348\end{gathered}$$"]
  end

  HG --> HGene
  Tok --> HGene
  Tok --> HRec
  CZ --> HRec
  HGene --> Std
  HRec --> Std
  Std --> Bins
  Bins --> Arms
  Bins --> Splits
  Lin --> Splits
  CZ -.->|"$$\text{frozen}$$"| Probe

  classDef sites fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef heads fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef loss fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef eval fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px

  class HG,CZ,Tok sites
  class HGene,HRec heads
  class Std,Bins,Arms loss
  class Lin,Splits,Probe eval
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

## 2026.10.20 - Review pass: heads per level, tokens at the head, binned loss

Redrawn for the 2026-10-20 review pass ([[experiments.043-expression-proteome-morphology]],
"Review pass"), with at most three boxes per row so the figure prints at or above 5 pt in
the primer. Two heads by readout level (gene-level over the 11 gene-level training
datasets, record-level for Vaishnav) replace the six heads by type, which become the
control arm; the type (ontology path), environment and dataset tokens are read at the head
with a per-dataset bias $c_d$; targets are standardized per dataset on training rows and the
main loss is HL-Gauss binned prediction (Farebrother et al. 2024), with the pinball and one
MSE cell as arms; the evaluation starts with the CPU linear ladder, reads splits A to E from
one fold file under the pre-registered rule, and keeps Ohya as a probe of the frozen latent.
Probe bars: 0.149 (019 genotype-only head) and 0.348 (Kemmeren ridge, job 5892). The
paragraph above about the numbers in the boxes describes the 2026-10-19 drawing.
