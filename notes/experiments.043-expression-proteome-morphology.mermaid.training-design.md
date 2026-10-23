---
id: e9lrxgy4aq03fx8a5xfv6z0
title: Training Design
desc: ''
updated: 1792755750398
created: 1792755750398
---

## 2026.10.23 - Data and model flow of the 043 training setup

Figure 1 of [[notes-tex/multimodal/043-training-design|dendron://torchcell/experiments.043-expression-proteome-morphology]]
(`notes-tex/multimodal/043-training-design/`). Top to bottom: the raw store, the input
representation, the fold roles, the prepared store, the sampler and batch, the model, the
heads, the losses, and the registered-window evaluation. Symbols follow the notation table of
that document, which reuses the Nature Biotechnology manuscript's symbols. Sources: record
counts from `results/verify_representation_043.json` (job 6120) and the primer; fold counts
from `results/global_split_v1_summary.json` (job 6046); prepared-store counts (50,767
training records, 784 reference contexts = S288C plus the 783 non-test isolates) from
`train_043_prep/prep_v1_routeA/meta.json` (job 6134); model and loss settings from
`conf/train_043_default.yaml`, `conf/loss/*.yaml` and `torchcell/models/cell_model_043.py`.
Colors follow [[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]]:
purple = data, orange = input side, red = model, yellow = objective and evaluation.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93','fontFamily':'Arial'}, 'flowchart': {'nodeSpacing': 14, 'rankSpacing': 26, 'subGraphTitleMargin': {'top': 6, 'bottom': 10}}}}%%
graph TD
  Store["$$\begin{gathered}\text{Raw 043 store}\\\ 15\ \text{datasets},\ 173{,}617\ \text{records}\\\ \text{record} = (\text{genotype},\ \varepsilon,\ y)\end{gathered}$$"]

  subgraph Rep["$$\text{Input representation (keys, not vectors)}$$"]
    direction LR
    Tok["$$\begin{gathered}\text{Composite token per gene } g_i,\ i \le N = 6{,}607\\\ \text{key per block } q \in \{\mathrm{up}, \mathrm{down}, \mathrm{prot}, \mathrm{calm}\}\\\ \text{or a typed gap of kind } \kappa\ (12\ \text{kinds})\\\ \text{position: } \mathrm{chr}(i),\ \xi_i,\ \nu^{\mathrm{chr}}_i,\ \nu^{\mathrm{gene}}_i\end{gathered}$$"]
    Chg["$$\begin{gathered}\text{Marks } t = 1, \ldots, M_b\\\ \text{change type } \tau_t\ (7\ \text{types})\\\ \text{precedence: absent} \succ \text{deletion} \succ \text{lof} \succ \text{variant}\\\ \text{allele parent } o_t\ \text{(segregants)}\end{gathered}$$"]
    Ctx["$$\begin{gathered}\text{Route A reference context } c\\\ \text{deletion, segregant, promoter: } c = \text{S288C}\\\ \text{isolate: } c = \text{the isolate's own alleles}\\\ \text{marks of an isolate: the HO deletion}\end{gathered}$$"]
  end

  Split["$$\begin{gathered}\text{Global split v1, fold roles}\\\ \text{genes: 10 folds, stratified by dataset membership}\\\ \text{isolates 5, segregants 5 within cross, promoters 5}\\\ \text{fold 0 test (never loaded)},\ \text{fold 1 validation},\ \text{folds} \ge 2\ \text{train}\\\ \text{Sameith Dm: split B};\ \text{Ohya: probe};\ \text{Sameith Sm: excluded}\end{gathered}$$"]

  Prep["$$\begin{gathered}\text{Prepared store (job 6134)}\\\ \text{test records dropped},\ y \mapsto \mathrm{tr}_s(y)\\\ 50{,}767\ \text{training records},\ 784\ \text{contexts}\\\ z = (\mathrm{tr}_s(y) - \eta_{s,i} - \mu_s) / \sigma_s\ \text{(training rows)}\end{gathered}$$"]

  Batch["$$\begin{gathered}\text{Sampler: batches of } B = 8\ \text{records}\\\ \text{records: uniform over training records}\\\ \text{balanced: dataset uniform, then record}\\\ \text{each distinct context encoded once}\end{gathered}$$"]

  subgraph Model["$$\text{Model: trunk, operator, heads}$$"]
    direction LR
    Trunk["$$\begin{gathered}\text{Trunk } F_\theta:\ H^{(0)} = (x_{\mathrm{CLS}}, x_1, \ldots, x_N)\\\ L = 4,\ d = 144,\ h = 9\\\ \text{graph mask at layer index 1, one } A^{(k)} \text{ per head}\\\ H_c = F_\theta(G_c) \in \mathbb{R}^{(N+1) \times d}\end{gathered}$$"]
    Op["$$\begin{gathered}\text{Perturbation operator}\\\ \text{queries: } h_{\mathrm{CLS}}, h_1, \ldots, h_N\\\ \text{keys: null key } \mathbf{k}_\varnothing \text{ and marks } \mathbf{p}_1, \ldots, \mathbf{p}_{M_b}\\\ \text{ReZero gates } \zeta_{\mathrm{attn}}, \zeta_{\mathrm{ffn}} = 0 \text{ at start}\end{gathered}$$"]
    Heads["$$\begin{gathered}\text{Heads (decoder)}\\\ \text{gene: } \mathrm{MLP}([h^{\mathrm{pert}}_{b,i}; \mathbf{c}_b]) + \mathbf{b}^{\mathrm{gene}}_{s(b)}\\\ \text{record: } \mathrm{MLP}([h^{\mathrm{pert}}_{b,\mathrm{CLS}}; \bar{\mathbf{k}}_b; \mathbf{c}_b]) + \mathbf{b}^{\mathrm{rec}}_{s(b)}\\\ \mathbf{c}_b = [\mathbf{c}^{\mathrm{type}}; \mathbf{c}^{\mathrm{data}}; \mathbf{c}^{\mathrm{env}}]\end{gathered}$$"]
  end

  subgraph Obj["$$\text{Objective}$$"]
    direction LR
    Loss["$$\begin{gathered}\text{Elementwise loss on measured } z\\\ \text{HL-Gauss: } n_{\mathrm{bin}} = 32 \text{ on } [-4, 4],\ \sigma_{\mathrm{HL}} = 0.75\,\Delta\\\ \text{pinball: } n_q = 19\ \text{quantiles}\\\ \text{MSE reference}\end{gathered}$$"]
    Wt["$$\begin{gathered}\text{per-dataset loss: mean over measured entries}\\\ \text{objective: weighted sum over datasets in the batch}\\\ \text{equal: } w_s = 1 / (\text{datasets present})\\\ \text{records: } w_s = (\text{records of } s) / B\end{gathered}$$"]
  end

  Eval["$$\begin{gathered}\text{Evaluation on fold 1, every epoch (never selects)}\\\ \text{registered window: last 20\% of epochs}\\\ \text{transfer} = \text{joint} - \text{alone at } n_{\mathrm{step}} = 317{,}300\\\ \text{rule: gain} \ge 0.02 \text{ in 5 of 6 seeds, strain-resampled CI}\end{gathered}$$"]

  Store --> Rep
  Tok ~~~ Chg ~~~ Ctx
  Rep --> Split
  Split --> Prep
  Prep --> Batch
  Batch --> Model
  Trunk --> Op
  Op --> Heads
  Model --> Obj
  Loss --> Wt
  Obj --> Eval

  classDef data fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef inp fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef model fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef obj fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  class Store,Split,Prep,Batch data
  class Tok,Chg,Ctx inp
  class Trunk,Op,Heads model
  class Loss,Wt,Eval obj
```

Render (PDF, outlined SVG and PNG land in `notes/assets/pdf-output/` under this note's name):

```bash
bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.043-expression-proteome-morphology.mermaid.training-design.md "#F5EEDD"
```
