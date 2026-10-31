---
id: swzdlb5sut1yzeutsb9yuer
title: Architecture Mermaid
desc: ''
updated: 1793127024348
created: 1793127024348
---

## 2026.10.27 - The two essentiality models drawn as mermaid diagrams with KaTeX

Author (2026-10-27, on Fig A.5): "we want more on model architecture... how big are these models,
num layers, if it can literally be drawn we should... or a mermaid diagram with latex, is also a
nice abstraction lets try both." This note is the mermaid abstraction, rendered by
`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.028-gene-essentiality.architecture-mermaid.md`
to `notes/assets/pdf-output/experiments.028-gene-essentiality.architecture-mermaid-1.pdf` (the CGT)
and `-2.pdf` (the GCN). The literal module graph is `draw_cell_architecture.py` (torchview).
Every number is from the w49 cell's result file (`n_parameters`) and `graph_baselines.py`. Styled
like the Type I / Type II diagram (author, 2026-10-27: "the mermaid should be in our nat biotech style"):
the beige canvas and the fill / border pairs of
[[torchcell.models.equivariant_cell_graph_transformer.mermaid.colors]], inputs purple, embeddings
and the deletion operator orange, message passing red, the graph prior and the readout yellow,
the output red.

### The CGT of Fig 3.1 (w49: one open block, 22,773 parameters)

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph Inputs["$$\text{Input layer: the genes}$$"]
    X["$$\begin{gathered}\text{every gene } i = 1, \dots, N\ (N = 6{,}607):\\\ x_i = [\,\text{pathway membership}\ (173)\ \|\ \text{ProtT5}\ (1024)\,] \in \mathbb{R}^{1197}\end{gathered}$$"]
  end
  X --> P["$$\begin{gathered}\textbf{preprocessor and token stack}\qquad \text{in: } x_1, \dots, x_N\\\ h_i^{(0)} = \mathrm{LN}(W_0 x_i + b_0) \in \mathbb{R}^{12}\\\ \text{out: } H^{(0)} = [\,c;\ h_1^{(0)};\ \dots;\ h_N^{(0)}\,] \in \mathbb{R}^{(N+1)\times 12},\\\ c \text{ a learned class token};\ 14{,}400 + 12\ \text{parameters}\end{gathered}$$"]
  P --> B1["$$\begin{gathered}\textbf{transformer block}\ (\text{4 heads, head width 3})\qquad \text{in: } H^{(0)}\\\ \alpha^{(k)} = \mathrm{softmax}\!\left(Q_k K_k^{\top}/\sqrt{3}\right):\ \text{every token attends to all } N+1\\\ Z = \mathrm{LN}\!\left(H^{(0)} + W_O[\,\alpha^{(1)}V_1;\ \dots;\ \alpha^{(4)}V_4\,]\right)\\\ H^{(1)} = \mathrm{LN}(Z + \mathrm{FFN}(Z))\\\ \text{out: } H^{(1)} \in \mathbb{R}^{(N+1)\times 12};\ 1{,}884\ \text{parameters}\\\ \text{no graph prior here (the KL pull or the mask act on a second block)}\end{gathered}$$"]
  B1 ~~~ S["$$\text{input: the strain, one deleted gene } g,\ S = \{g\}$$"]
  S --> OP["$$\begin{gathered}\textbf{deletion operator}\ (\text{cross-attention, 6 heads})\\\ \text{in: } H = H^{(1)}_{1:N}\ \text{(one row per gene)},\ S\\\ \text{gene } i \text{ queries the deleted gene: } c_i = \mathrm{Attn}(h_i;\ \{h_p : p \in S\})\\\ |S| = 1 \Rightarrow c_i = W_O(W_V h_g + b_V) + b_O,\ \text{the same for every } i\\\ Z_i = \mathrm{LN}(h_i + c_i),\qquad h_i^{\mathrm{pert}} = \mathrm{LN}(Z_i + \mathrm{FFN}(Z_i))\\\ \text{out: } H^{\mathrm{pert}} \in \mathbb{R}^{N \times 12};\ 1{,}884\ \text{parameters}\end{gathered}$$"]
  B1 --> OP
  OP ~~~ U["$$\begin{gathered}\text{input: the metabolic solve of the deletion of } g\\\ u_g = [\,r_g;\ \log(1 + |v_g|)\,] \in \mathbb{R}^{4135}\\\ r_g \in [0,1]^4 \text{ the FBA growth ratios on four media,}\\\ v_g \in \mathbb{R}^{4131} \text{ the pFBA fluxes on YPD}\end{gathered}$$"]
  U --> RD["$$\begin{gathered}\textbf{readout}\qquad \text{in: } h_{\mathrm{CLS}} = H^{(1)}_0\ \text{(the reference cell)},\ H^{\mathrm{pert}},\ S,\ u_g\\\ z = [\,h_{\mathrm{CLS}};\ \tfrac{1}{N}\textstyle\sum_i h_i^{\mathrm{pert}};\ \tfrac{1}{|S|}\textstyle\sum_{p \in S} h_p^{\mathrm{pert}}\,] \in \mathbb{R}^{36}\\\ \ell_g = w_2^{\top}\mathrm{ReLU}(W_1 z + b_1) + b_2 \;+\; w^{\top} u_g + b\\\ \text{out: } \ell_g \in \mathbb{R};\ 457 + 4{,}136\ \text{parameters}\end{gathered}$$"]
  OP --> RD
  RD --> Y["$$\begin{gathered}\text{in: } \ell_g;\ \text{out: essential if } \ell_g > 0\\\ \text{class-weighted BCE on 629 training genes, 200 epochs of AdamW;}\\\ \text{read at the best validation epoch, scored on 195 held-out genes}\end{gathered}$$"]
  classDef input fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef embedding fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef transformer fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef typeI fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef invariant fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef output fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  class X,S,U input
  class P embedding
  class B1 transformer
  class OP typeI
  class RD invariant
  class Y output
```

### The GCN of Fig 3.1 (GO input, 155,457 parameters)

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'}}}%%
graph TD
  subgraph Inputs2["$$\text{Input layer: the genes and the graph}$$"]
    GO["$$\begin{gathered}\text{every gene } i = 1, \dots, N\ (N = 6{,}607):\ x_i \in \{0,1\}^{2362},\\\ \text{the GO terms with ten or more genes, propagated}\\\ \text{through is-a and part-of, all three namespaces}\end{gathered}$$"]
    E["$$\begin{gathered}\text{the edges of the union of the nine gene graphs:}\\\ A \in \{0,1\}^{N \times N},\quad \hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2},\\\ \tilde{D} \text{ the degree matrix of } A + I\end{gathered}$$"]
    GO ~~~ E
  end
  E ~~~ C1
  GO --> C1["$$\begin{gathered}\textbf{graph convolution 1}\qquad \text{in: } X = [\,x_1;\ \dots;\ x_N\,],\ \hat{A}\\\ H^{(1)} = \mathrm{ReLU}(\hat{A} X W_1 + b_1),\ \text{then dropout } 0.2\\\ \text{out: } H^{(1)} \in \mathbb{R}^{N \times 64};\ 151{,}232\ \text{parameters}\end{gathered}$$"]
  E --> C1
  C1 --> C2["$$\begin{gathered}\textbf{graph convolution 2}\qquad \text{in: } H^{(1)},\ \hat{A}\\\ H^{(2)} = \mathrm{ReLU}(\hat{A} H^{(1)} W_2 + b_2)\\\ \text{out: } H^{(2)} \in \mathbb{R}^{N \times 64};\ 4{,}160\ \text{parameters}\end{gathered}$$"]
  E --> C2
  C2 --> L["$$\begin{gathered}\textbf{readout}\qquad \text{in: } H^{(2)}\\\ \ell_i = w^{\top} h_i^{(2)} + b\ \text{for every gene at once}\\\ \text{out: } \ell_1, \dots, \ell_N;\ 65\ \text{parameters}\end{gathered}$$"]
  L --> Y2["$$\begin{gathered}\text{in: } \ell_i;\ \text{out: essential if } \ell_i > 0\\\ \text{one pass over the whole graph: class-weighted BCE on the}\\\ \text{629 training nodes, 400 epochs of Adam; read at the best}\\\ \text{validation epoch; the held-out genes are nodes of the same graph}\end{gathered}$$"]
  classDef input fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef transformer fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef invariant fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef output fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  class GO,E input
  class C1,C2 transformer
  class L invariant
  class Y2 output
```

Function view (author, 2026-10-27: "each next should contain outputs of previous box so it
is clear. we want the model as function view"): every box names what it takes in, the map,
what it puts out, and its parameter count, and the output symbol of one box is the input
symbol of the next. Self-contained: no reference to another figure inside a diagram.

#### Coherence audit against the code (2026-10-27)

Checked against `torchcell/models/equivariant_cell_graph_transformer.py`
(`GraphRegularizedTransformerLayer.forward`, `EquivariantPerturbationTransform.forward`),
`torchcell/models/cell_graph_transformer_metabolism.py` (`ProductScalarHead`,
`CellGraphTransformerMetabolism.forward`), `train_essentiality.py` (`FbaAffine`,
`FbaContextReadout`, `_select_epoch`, `fcl_train_pos_weight`, the solve encodings) and
`graph_baselines.py` (`Net.forward`, `run_arm`, `smoothed_argmax`), plus the w49 result
file `w49_pathfbapfbalogfree_h12l1p1_e200_s2700.json` (the stem `pattern_ladder_readout.json`
reports as w49) and the torchview drawing of the same cell. Two statements of the first
version were wrong and are corrected above:

- The deletion operator ran the other way round. The code has every gene token as the
  query and the deleted gene's token as the one key and value, so with one deletion the
  attended context is one vector shared by all genes, and the transform returns all N
  perturbed rows; the readout then pools them. The first version wrote the deleted gene as
  the query over the gene rows.
- The affine over the solve is free-initialized in w49 (`ess_solve_zero_init` false) and
  reads the pFBA vector as `log(1 + |v|)` (`ess_solve_encoding` log1p); the zero-init
  recipe is the `pfbazero` stem, whose seed-2700 cell scores 0.80, not the figure's 0.91.
- Also fixed: the readout takes three 12-vectors (class token, mean over all perturbed
  rows, mean over the deleted rows), 36 wide, not two; the loss runs on the 629 training
  genes (158 validation, 195 held out), not 787; the graph prior is inert here because its
  head map targets the second block and this cell has one.

Confirmed as written: post-LN in the block and in the operator, 4 heads of width 3,
FFN 12 to 48 to 12, every parameter count (preprocessor 14,400; class token 12; block
1,884; operator 1,884; readout 457 + 4,136 = 4,593; total 22,773), the GCN's two
convolutions with ReLU and dropout 0.2 between, its 2,362 GO columns and 155,457 parameters,
the 400-epoch arm (`graph_baselines_go400_s6.json`) as the one Fig 3.1 reports.

#### Layout (2026-10-27, later)

Both diagrams are single columns so that one shared scale prints them side by side at 165 mm
with text at 5.6 pt: the strain and the metabolic solve enter as purple input boxes placed
directly above the operator and the readout that consume them (invisible mermaid links
`~~~` pin their rank), because a side box or a long edge from the top makes dagre stagger
the whole column into a staircase twice as wide. Training details that do not change the
map (batch size, learning rate, weight decay, the class weights, the epoch rule) live in the
symbol table, not in the boxes. Composed by `compose_architecture_figure.py`
([[experiments.028-gene-essentiality.scripts.compose_architecture_figure]]).

#### Bold in KaTeX labels (2026-10-27, later)

Author: "can you use bold in these? it could help styling." `\textbf{}` was in the labels
but printed regular: mermaid renders KaTeX to MathML, and Chromium's MathML Core honors
`mathvariant="normal"` only, so bold and bold-italic were dropped at print. The render
script now passes `notes/assets/publish/scripts/mermaid_mathml.css` to mmdc, which maps
`mathvariant="bold"` and `"bold-italic"` to CSS weight and style and sets
`foreignObject { overflow: visible }` so no label glyph is clipped at its box edge
(Liberation Serif Bold is now embedded in the PDF). Every mermaid diagram rendered through
the script gets this, so an older diagram with `\textbf` in a label changes on its next
render.

## 2026.10.31 - Recovered to main, re-read against main's model code

This note, `notes/assets/publish/scripts/mermaid_mathml.css`, the `-C` line of
`mermaid_pdf.sh` and the two renders were untracked in worktree `docs/essentiality-figure-gate`
and in no commit anywhere (the thesis sheet `fig-ch03-essentiality-architecture.pdf` was composed
from them on 2026-10-27). They land on main through branch `fig/cgt-architecture-mermaid` so
the diagram can go into the manuscript's supplementary information beside the trigenic and the
multi-measurement models ([[experiments.025-solid-growth.architecture-mermaid]],
[[experiments.030-solid-growth-multi.architecture-mermaid]]). `compose_architecture_figure.py`
stays in that worktree with the rest of `experiments/028-gene-essentiality/`, which is not on
main; the manuscript sheet is composed by `paper/nature-biotech/scripts/compose_cgt_architecture_si.py`.

Re-read against main at `3d3759b58`: the post-norm block with the 4x feed-forward
(`GraphRegularizedTransformerLayer.forward`), the operator with every gene as a query over
the deleted gene's row as the one key and value (`EquivariantPerturbationTransform.forward`),
and the readout's three 12-wide pools (`ProductScalarHead.forward` in
`cell_graph_transformer_metabolism.py`: the class token, the mean over all perturbed rows,
the mean over the deleted rows) are as drawn. The solve affine, the training script and the
w49 result file are on the 028 branch only; those statements are unchanged from the
2026-10-27 audit above and were not re-run here.
