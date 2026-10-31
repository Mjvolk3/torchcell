---
id: 3yvnostwtbjq6x1nwu0zybh
title: Architecture Mermaid
desc: ''
updated: 1793431572335
created: 1793431572335
---

## 2026.10.31 - The multi-measurement Cell Graph Transformer with the experiment ontology as its prior

The model of thesis Section 3.7 (Fig. 3.10), configuration
`experiments/030-solid-growth-multi/conf/cgt_030_s3_r_onto_affine_010.yaml` on branch
`feat/030-token-affine` (chain `_010 -> _tok_embfit_003 -> _tok_fit_002 -> _tok_fit_000`):
the trigenic model of [[experiments.025-solid-growth.architecture-mermaid]] trained on the S3
closure pool (every stored single, double and triple measurement, 1,121,662 records of 15
sources) with four changes. The gene table is replaced by a frozen composite sequence embedding
through a parameter-matched preprocessor; the class token rides through the deletion operator
and a linear fitness head reads it; every stored measurement is its own training row; and the
row's experiment annotation on the schema ontology, propagated to the root by a DCell-style
encoder, sets a scale and shift on each head's output (`dataset_token: {mode: affine, source:
ontology}`), so the model is conditioned on what the experiment is and never on which dataset
reported it. Same function view and palette as the 025 diagram; rendered by
`bash notes/assets/publish/scripts/mermaid_pdf.sh notes/experiments.030-solid-growth-multi.architecture-mermaid.md`.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#F5EEDD','clusterBkg':'#F5EEDD','clusterBorder':'#E0D6BE','lineColor':'#B7AC93'},'flowchart':{'rankSpacing':26,'nodeSpacing':20,'padding':6}}}%%
graph TD
  subgraph Inputs["$$\text{Input layer: the genes}$$"]
    X["$$\begin{gathered}\text{every gene } i = 1, \dots, N\ (N = 6{,}607):\ \text{a frozen composite sequence embedding}\\\ x_i = [\,\text{SpeciesLM upstream};\ \text{CaLM};\ \text{ProtT5};\ \text{SpeciesLM downstream}\,] \in \mathbb{R}^{3328}\end{gathered}$$"]
  end
  X --> P["$$\begin{gathered}\textbf{preprocessor and token stack}\qquad \text{in: } x_1, \dots, x_N\\\ e_i = \mathrm{LN}\!\big(W_2\,\mathrm{GELU}(\mathrm{LN}(W_1 x_i + b_1)) + b_2\big) \in \mathbb{R}^{180},\ 3328 \to 339 \to 180\\\ \text{out: } H^{(0)} = [\,c;\ e_1;\ \dots;\ e_N\,] \in \mathbb{R}^{(N+1)\times 180},\ c \text{ a learned class token};\ 1{,}190{,}769 + 180\ \text{parameters}\end{gathered}$$"]
  P ~~~ G["$$\begin{gathered}\text{input: the nine gene graphs } A^{(k)} \in \{0,1\}^{N \times N},\ k = 1, \dots, 9\\\ \text{(physical, regulatory, TFLink, six STRING channels)},\\\ \tilde{A}^{(k)} = (D^{(k)})^{-1} A^{(k)} \text{ row-normalized; training only}\end{gathered}$$"]
  G -.-> B1
  P --> B1["$$\begin{gathered}\textbf{encoder}\ (L = 8\ \text{transformer blocks, 9 heads, head width 20})\qquad \text{in: } H^{(0)}\\\ \text{block } \ell:\ \alpha^{(\ell,k)} = \mathrm{softmax}\!\left(Q_k K_k^{\top}/\sqrt{20}\right):\ \text{every token attends to all } N+1\\\ Z = \mathrm{LN}\!\left(H^{(\ell-1)} + W_O[\,\alpha^{(\ell,1)}V_1;\ \dots;\ \alpha^{(\ell,9)}V_9\,]\right),\quad H^{(\ell)} = \mathrm{LN}(Z + \mathrm{FFN}(Z)),\ \mathrm{FFN}: 180 \to 720 \to 180\\\ \text{graph prior on block 2, head } k \text{ toward graph } k:\ \Omega_k = \textstyle\sum_{i} D_{\mathrm{KL}}\!\big(\tilde{A}^{(k)}_{i,:}\,\|\,\alpha^{(2,k)}_{i,:}\big)\ \text{over the gene rows}\\\ \text{out: } H = H^{(8)} \in \mathbb{R}^{(N+1)\times 180};\ 8 \times 391{,}140 = 3{,}129{,}120\ \text{parameters}\end{gathered}$$"]
  B1 ~~~ S["$$\text{input: the strain, its deleted genes } S,\ |S| \in \{1, 2, 3\}$$"]
  S --> OP["$$\begin{gathered}\textbf{deletion operator}\ (\text{cross-attention, 9 heads})\qquad \text{in: } H\ \text{(the class token and one row per gene)},\ S\\\ \text{every row } i \in \{\mathrm{CLS}, 1, \dots, N\} \text{ queries the deleted genes: } c_i = \mathrm{Attn}(h_i;\ \{h_p : p \in S\})\\\ Z_i = \mathrm{LN}(h_i + c_i),\qquad h_i^{\mathrm{pert}} = \mathrm{LN}(Z_i + \mathrm{FFN}(Z_i)),\ \mathrm{FFN}: 180 \to 720 \to 180\\\ \text{out: } h_{\mathrm{CLS}}^{\mathrm{pert}} \in \mathbb{R}^{180},\ H^{\mathrm{pert}} \in \mathbb{R}^{N \times 180};\ 391{,}140\ \text{parameters}\end{gathered}$$"]
  B1 --> OP
  OP --> RD["$$\begin{gathered}\textbf{readouts}\qquad \text{in: } h_{\mathrm{CLS}} = H_0,\ h_{\mathrm{CLS}}^{\mathrm{pert}},\ H^{\mathrm{pert}},\ S\\\ \text{interaction: } z = [\,h_{\mathrm{CLS}};\ \textstyle\sum_{p \in S} h_p^{\mathrm{pert}}\,] \in \mathbb{R}^{360},\quad \hat{y}_{\mathrm{int}} = w_2^{\top}\mathrm{ReLU}(W_1 z + b_1) + b_2;\ 65{,}161\ \text{parameters}\\\ \text{fitness: } \hat{y}_{\mathrm{fit}} = w^{\top} h_{\mathrm{CLS}}^{\mathrm{pert}} + b,\ \text{a linear probe of the perturbed cell state};\ 181\ \text{parameters}\\\ \text{out: } \hat{y}_{\mathrm{int}},\ \hat{y}_{\mathrm{fit}} \in \mathbb{R},\ \text{one pair per strain}\end{gathered}$$"]
  RD ~~~ U["$$\begin{gathered}\text{input: one row } e \text{ per stored measurement of the strain, with its annotation on the}\\\ \text{experiment ontology } a_e \in \mathbb{R}^{149}:\ \text{the experiment, phenotype and reference classes and their}\\\ \text{parents, the medium's components, dropouts, state and base, aerobicity, the temperature unit}\\\ \text{and value (celsius over 100), the origin class of a converted record; the dataset name is left out}\end{gathered}$$"]
  U --> ON["$$\begin{gathered}\textbf{ontology prior}\ (\text{DAG encoder and affine calibration})\qquad \text{in: } a_e,\ \hat{y}_{\mathrm{int}},\ \hat{y}_{\mathrm{fit}}\\\ \text{every node } v, \text{ leaves first (depth 4): } s_v = \tanh\!\Big(\textstyle\sum_{u \in \mathrm{children}(v)} s_u W_u + [\,a_{e,v};\ \mathbb{1}[a_{e,v} \neq 0]\,]\,U_v + b_v\Big)\ \text{if } v \text{ annotated or above one, else } 0\\\ \text{token } t_e = W_{\mathrm{out}}\, s_{\mathrm{Experiment}} + b_{\mathrm{out}} \in \mathbb{R}^{8};\ \ [\,\Delta\alpha_{\mathrm{int}};\ \Delta\beta_{\mathrm{int}};\ \Delta\alpha_{\mathrm{fit}};\ \Delta\beta_{\mathrm{fit}}\,] = W_{\mathrm{aff}}\, t_e + b_{\mathrm{aff}}\ \text{(zero at initialization)}\\\ \text{out: } y_{e,\mathrm{int}} = (1 + \Delta\alpha_{\mathrm{int}})\,\hat{y}_{\mathrm{int}} + \Delta\beta_{\mathrm{int}},\quad y_{e,\mathrm{fit}} = (1 + \Delta\alpha_{\mathrm{fit}})\,\hat{y}_{\mathrm{fit}} + \Delta\beta_{\mathrm{fit}};\ 13{,}184 + 36\ \text{parameters}\end{gathered}$$"]
  RD --> ON
  ON --> Y["$$\begin{gathered}\text{in: } y_{e,\mathrm{int}},\ y_{e,\mathrm{fit}};\ \text{out: the interaction and the fitness of row } e\ \text{(standardized, one scale per label over every source)}\\\ \mathcal{L} = \mathrm{MSE}_{\mathrm{int}} + 0.1\, W_2 + \mathrm{MSE}_{\mathrm{fit}} + \textstyle\sum_{k=1}^{9} \lambda_k \tfrac{N}{|E_k|}\,\Omega_k,\ \lambda_k = 10^{-3},\ \text{each term over the rows that carry the label}\\\ \text{1,046,316 training rows of 15 sources; AdamW at } 2.5 \times 10^{-4}, \text{ 200 epochs, four GPUs, 256 strains each}\end{gathered}$$"]
  classDef input fill:#E1D5E7,stroke:#846592,stroke-width:2px
  classDef embedding fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef transformer fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  classDef typeI fill:#FFE6CC,stroke:#BD8800,stroke-width:2px
  classDef invariant fill:#FFF2CC,stroke:#BCA04C,stroke-width:2px
  classDef output fill:#F8CECC,stroke:#A24A46,stroke-width:2px
  class X,S,G,U input
  class P embedding
  class B1 transformer
  class OP typeI
  class RD,ON invariant
  class Y output
```

### Coherence audit against the code (2026.10.31)

Checked against branch `feat/030-token-affine` at `3c41725c4`:
`torchcell/models/equivariant_cell_graph_transformer.py` (`CellGraphTransformer.__init__`, the
preprocessor with `hidden_dim`, `dataset_token` and `perturb_cls`; `forward`, `entry_heads`,
`_entry_calibrate`; `PerturbationHead.combined` and `forward_entries`; `GlobalHead` with
`linear` and `use_gene_pool=False`), `torchcell/data/experiment_ontology.py`
(`ExperimentOntologyVocabulary.annotate`, `OntologyDagEncoder._propagate`),
`torchcell/trainers/int_transformer_cell.py` (`_shared_step_per_entry`, the fitness term),
`experiments/030-solid-growth-multi/scripts/equivariant_cell_graph_transformer.py` (the
vocabulary build and the model call) and the config chain. Parameter counts are the model's
own tally on an instance built from the config with the vocabulary the run wrote
(`results/experiment_ontology_vocab_030.json`, 149 nodes, depth 4) and a 3,328-wide input:
preprocessor 1,190,769 (parameter-matched to the 6,607 x 180 table it replaces, 1,189,260);
class token 180; encoder 3,129,120; operator 391,140; interaction head 65,161; fitness head
181; ontology encoder 13,184 (`child_proj` 149 x 8 x 8, `own_proj` 149 x 2 x 8, `bias`
149 x 8, `out` 8 x 8 + 8) and the affine map 36 (`entry_affine_proj`, 8 to 4); total
4,789,771. What the code fixes:

- `perturb_cls: true` sends the class token through the operator as query row 0 over the
  same deleted-gene keys; the gene rows are bit-identical to the 025 path. The interaction
  head still reads the wild-type class token (`perturbation_head_cls: wildtype`), the fitness
  head reads the perturbed one (`heads.global: {decoder: s1_pool, use_gene_pool: false,
  linear: true}`), so the fitness head is `w . h_CLS_pert + b`.
- In affine mode nothing about the source enters the heads (`token` is an empty tensor);
  the ontology token acts on the head outputs through `entry_affine_proj`, zero-initialized
  so training starts at the identity calibration, with both the scale and the shift free
  (`scale: true, shift: true`). The readout variant (`_011`, `mode: readout`) instead appends
  the 8-wide token to both head inputs (360 + 8 and 180 + 8) and has no calibration.
- The encoder is DCell-style over the vocabulary's DAG: each child projects its own state
  toward its parent with its own matrix (`child_proj[u]`), the parent sums the projections
  (a sum, not a mean), adds its own annotation through `own_proj[v]` over `[a_v, 1[a_v != 0]]`
  and a bias, applies tanh, and is zeroed unless it is annotated or above an annotated node
  (`active`); computed one depth at a time, leaves first; the token is `out(s_root)`.
- The annotation (`annotate`) marks the experiment class and its inheritance chain to the
  `Experiment` root, the phenotype class and its chain, `Environment` with the aerobicity
  value node, `Media` with its component, dropout, state and base-medium value nodes,
  `Temperature` with its unit node and `Temperature.value` = celsius / 100, the reference
  classes under `ref:`, and `Experiment.converted_from:<class>` for the SGD essentiality and
  SynthLethDB records; `dataset_name` is excluded by construction. The 15 sources of S3 fall
  into 8 distinct annotation profiles (the two Kuzmin screens share one).
- Rows: the processor emits one row per stored measurement with `phenotype_values_batch`
  naming its strain and `phenotype_ontology` its annotation; the interaction loss is the
  025 `PointDistGraphReg` (MSE, Wasserstein at 0.1, the graph term) over the rows that carry
  an interaction, and the fitness term is `fitness_lambda` (1.0) times the MSE over the rows
  that carry a fitness. Labels are standardized with one mean and sd per label over the
  whole pool (`normalization_stats_030_s3_r.json`), never per source.
- Optimizer: AdamW at a constant 2.5e-4 (`lr_scheduler: null`), 200 epochs, bf16-mixed, four
  A100s at batch 256 per card; the R split (random over triples) with the single and double
  rows unpinned to training.

![](./assets/pdf-output/experiments.030-solid-growth-multi.architecture-mermaid.svg)
