---
id: tlumf7mjp00dwckyowfnucb
title: Architecture Mermaid
desc: ''
updated: 1793431564151
created: 1793431564151
---

## 2026.10.31 - The trigenic Cell Graph Transformer drawn as a function

The model of the trigenic result (Fig. 2 of the manuscript, thesis Fig. 3.5d), configuration
`experiments/025-solid-growth/conf/cgt_s0_r_kl_000.yaml` (the 010 model copied verbatim onto the
025 build: hidden width 180, eight encoder blocks of nine heads, the KL prior on the nine heads
of the second block, a nine-head cross-attention deletion operator, the interaction readout).
Drawn in the function view of [[experiments.028-gene-essentiality.architecture-mermaid]]: every
box names what it takes in, the map, what it puts out and its parameter count, and the output
symbol of one box is the input symbol of the next. Rendered by
`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.025-solid-growth.architecture-mermaid.md`
to `notes/assets/pdf-output/experiments.025-solid-growth.architecture-mermaid.pdf`. Colors are the
Type I / Type II palette of [[torchcell.models.equivariant_cell_graph_transformer.mermaid.colors]]:
inputs purple, the embedding and the deletion operator orange, the attention blocks and the output
red, the graph prior and the readout yellow.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'},'flowchart':{'rankSpacing':26,'nodeSpacing':20,'padding':6}}}%%
graph TD
  subgraph Inputs["$$\text{Input layer: the genes}$$"]
    X["$$\begin{gathered}\text{every gene } i = 1, \dots, N\ (N = 6{,}607):\\\ \text{a learned row } e_i \in \mathbb{R}^{180} \text{ of the gene table } E \in \mathbb{R}^{N \times 180};\ 1{,}189{,}260\ \text{parameters}\end{gathered}$$"]
  end
  X --> P["$$\begin{gathered}\textbf{token stack}\qquad \text{in: } e_1, \dots, e_N\\\ \text{out: } H^{(0)} = [\,c;\ e_1;\ \dots;\ e_N\,] \in \mathbb{R}^{(N+1)\times 180},\\\ c \text{ a learned class token};\ 180\ \text{parameters}\end{gathered}$$"]
  P ~~~ G["$$\begin{gathered}\text{input: the nine gene graphs } A^{(k)} \in \{0,1\}^{N \times N},\ k = 1, \dots, 9\\\ \text{(physical, regulatory, TFLink, six STRING channels)},\\\ \tilde{A}^{(k)} = (D^{(k)})^{-1} A^{(k)} \text{ row-normalized; training only}\end{gathered}$$"]
  G -.-> B1
  P --> B1["$$\begin{gathered}\textbf{encoder}\ (L = 8\ \text{transformer blocks, 9 heads, head width 20})\qquad \text{in: } H^{(0)}\\\ \text{block } \ell:\ \alpha^{(\ell,k)} = \mathrm{softmax}\!\left(Q_k K_k^{\top}/\sqrt{20}\right):\ \text{every token attends to all } N+1\\\ Z = \mathrm{LN}\!\left(H^{(\ell-1)} + W_O[\,\alpha^{(\ell,1)}V_1;\ \dots;\ \alpha^{(\ell,9)}V_9\,]\right),\quad H^{(\ell)} = \mathrm{LN}(Z + \mathrm{FFN}(Z)),\ \mathrm{FFN}: 180 \to 720 \to 180\\\ \text{graph prior on block 2, head } k \text{ toward graph } k:\ \Omega_k = \textstyle\sum_{i} D_{\mathrm{KL}}\!\big(\tilde{A}^{(k)}_{i,:}\,\|\,\alpha^{(2,k)}_{i,:}\big)\ \text{over the gene rows}\\\ \text{out: } H = H^{(8)} \in \mathbb{R}^{(N+1)\times 180};\ 8 \times 391{,}140 = 3{,}129{,}120\ \text{parameters}\end{gathered}$$"]
  B1 ~~~ S["$$\text{input: the strain, its deleted genes } S = \{g_1, g_2, g_3\}$$"]
  S --> OP["$$\begin{gathered}\textbf{deletion operator}\ (\text{cross-attention, 9 heads})\qquad \text{in: } H_{1:N}\ \text{(one row per gene)},\ S\\\ \text{gene } i \text{ queries the deleted genes: } c_i = \mathrm{Attn}(h_i;\ \{h_p : p \in S\}),\ \text{a softmax over the } |S| \text{ keys}\\\ Z_i = \mathrm{LN}(h_i + c_i),\qquad h_i^{\mathrm{pert}} = \mathrm{LN}(Z_i + \mathrm{FFN}(Z_i)),\ \mathrm{FFN}: 180 \to 720 \to 180\\\ \text{out: } H^{\mathrm{pert}} \in \mathbb{R}^{N \times 180};\ 391{,}140\ \text{parameters}\end{gathered}$$"]
  B1 --> OP
  OP --> RD["$$\begin{gathered}\textbf{interaction readout}\qquad \text{in: } h_{\mathrm{CLS}} = H_0\ \text{(the reference cell)},\ H^{\mathrm{pert}},\ S\\\ z = [\,h_{\mathrm{CLS}};\ \textstyle\sum_{p \in S} h_p^{\mathrm{pert}}\,] \in \mathbb{R}^{360}\\\ \hat{y} = w_2^{\top}\mathrm{ReLU}(W_1 z + b_1) + b_2\\\ \text{out: } \hat{y} \in \mathbb{R};\ 65{,}161\ \text{parameters}\end{gathered}$$"]
  RD --> Y["$$\begin{gathered}\text{in: } \hat{y};\ \text{out: the trigenic interaction } \hat{\tau}_{ijk}\ \text{(standardized)}\\\ \mathcal{L} = \mathrm{MSE}(\hat{y}, \tau) + 0.1\, W_2(\hat{y}, \tau) + \textstyle\sum_{k=1}^{9} \lambda_k \tfrac{N}{|E_k|}\,\Omega_k,\ \lambda_k = 10^{-3}\\\ \text{376,732 triples of two screens; AdamW, four GPUs, 256 strains each}\end{gathered}$$"]
  classDef input fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef embedding fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef transformer fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef typeI fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef invariant fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef output fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  class X,S,G input
  class P embedding
  class B1 transformer
  class OP typeI
  class RD invariant
  class Y output
```

### Coherence audit against the code (2026.10.31)

Checked against `torchcell/models/equivariant_cell_graph_transformer.py` on main
(`GraphRegularizedTransformerLayer.forward`, `EquivariantPerturbationTransform.forward`,
`PerturbationHead.forward`, `CellGraphTransformer.compute_graph_regularization_loss`,
`_normalize_adjacency_matrices`), `torchcell/losses/point_dist_graph_reg.py`
(`PointDistGraphReg.forward`), `experiments/025-solid-growth/scripts/equivariant_cell_graph_transformer.py`
(the model and loss construction) and `conf/cgt_s0_r_kl_000.yaml`. The parameter counts are the
model's own tally, `CellGraphTransformer.num_parameters` on an instance built from the config
over a 6,607-gene graph: gene table 1,189,260; class token 180; encoder 3,129,120 (each block:
Q, K, V and output projections 130,320, feed-forward 260,100, two norms 720); operator 391,140;
readout 65,161; total 4,774,861. Statements the code fixes, and that an earlier description got
wrong somewhere:

- The encoder is post-norm: `norm1(x + dropout(out_proj(attn)))` then `norm2(z + dropout(ffn(z)))`,
  with the feed-forward 180 to 720 to 180 and GELU. The thesis review of 2026-10-29 corrected the
  same point in Section 3.4.
- The operator runs on the gene rows only; the class token is not transformed, and the readout
  reads the reference cell's class token (`h_CLS = H[0]`). Every gene is a query and the
  deleted genes' encoded rows are the keys and values (`nn.MultiheadAttention`, nine heads).
- The readout sums the perturbed rows of the deleted genes (`pooling` defaults to `sum`; the
  config sets nothing, so the 025 runs sum where 010 averaged; at a constant three deletions
  the factor is absorbed by the first linear layer).
- The graph prior sits on `graph_reg_layer: 1`, the second block in reading order; head k of
  that block is pulled toward graph k (the nine `regularized_heads`); the KL runs over the
  gene-gene block with the class row and column excluded (`attention_weights[:, :, 1:, 1:]`),
  every row (`row_sampling_rate: 1.0`), and each graph's term is divided by its edge count over
  N (`kl_loss / (edges_k / gene_num)`) with `lambda_k = 1e-3` applied once.
- The loss is `PointDistGraphReg`: MSE at weight 1, the Wasserstein term at 0.1 (`geomloss`
  Sinkhorn, p = 2, blur 0.05, scaling 0.9; `use_buffer: false` selects the unbuffered
  `WeightedWassersteinLoss`, which has no minimum batch size, so the term is live on every
  batch of 256 although the config's `min_samples_for_wasserstein: 512` would gate the buffered
  version), and the model's graph term at weight 1.
- Labels are standardized over the 376,732 triples (`fit_on: subset`), so the readout predicts
  the standardized interaction.

![](./assets/pdf-output/experiments.025-solid-growth.architecture-mermaid.svg)
