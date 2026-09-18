# Agent 08: the inputs. What does the model see about each gene and about the graph?

2026-09-17/18 review round. Read-only. Every number below traces to a file, a run id, or a
script I ran in the scratchpad. New measurements I made in this session are marked
**[NEW]** and their scripts are named at the end; they are scratchpad scripts, not
committed artifacts, so they are evidence for a decision and not paper artifacts.

---

## 0. The short answer

The incumbent sees, per gene, a 3,328-dimensional concatenation of four frozen sequence
embeddings and nothing else; it sees the gene graph as a symmetric hard attention mask on
**one of six** encoder layers; and the deletion reaches every reporter through a single
90-dimensional vector.

Against that, the retrieval structure the task actually needs is present in the data and
invisible to every representation we have. **[NEW]** A neighbor average that retrieves
with the strain's own measured profile reaches per-feature Pearson **0.629 +/- 0.011
validation, 0.646 +/- 0.011 test** across the four v13 split seeds, while the same rule
retrieving with ProtT5 reaches **0.107 +/- 0.019 / 0.116 +/- 0.057** and the trained model
reaches 0.1965 +/- 0.0222. The gene representation, not the architecture and not the
objective, is what stands between 0.12 and 0.63.

The mechanism is measurable and blunt. **[NEW]** Over the 703,230 pairs of deleted genes
in the split-seed-0 training rows, 5.48% of pairs have deletion-profile correlation above
0.3 and the 99th percentile is 0.457, so the neighbor structure is real. The Spearman
correlation between embedding cosine similarity and that profile correlation is **+0.017
for ProtT5, +0.014 for CaLM, +0.012 for the chromatin-pathway vector and +0.018 for a
train-split co-expression vector**. No available gene *node feature* aligns with
deletion-response similarity at more than rho = 0.02.

**[NEW] Two of the nine graphs do, and they are not the ones the campaign has been
reasoning about.** Genes joined on `string12_0_cooccurence` (phylogenetic co-occurrence) or
`string12_0_database` (STRING's curated pathway and complex channel) are **4.8x and 3.6x**
more likely than base rate to have correlated deletion profiles, and **8.5x and 13.0x** at
the r > 0.5 threshold, while the two TF-target graphs sit at **0.98x and 0.85x**, i.e. at
chance. These two informative channels are precisely the low-coverage ones that the layer-1
mask leaves dark for 39% to 61% of genes.

---

## 1. What is in the token, with dimensions and measured marginal value

### 1.1 The resolved input of the incumbent

`cell_dataset.node_embeddings` resolves for `cgt_expr_v13_split` (and for v12, v11's
`E_full` arm, and v15) to

| slot | name | dim | source model |
|---|---|--:|---|
| 5' flank | `fudt_upstream` | 768 | species-aware fungal up/down transformer |
| ORF | `calm` | 768 | codon language model over the ORF |
| protein | `prot_T5_all` | 1,024 | ProtT5 XL UniRef50 |
| 3' flank | `fudt_downstream` | 768 | same fungal model, 3' window |
| | **total** | **3,328** | |

Set at `experiments/019-simb-multimodal/conf/cgt_expr_v11_emb.yaml` and inherited
unchanged. `learnable_embedding.enabled: false`, `gene_num: 6607`, `hidden_channels: 90`.
The input projection's hidden layer is `(input + 90) / 2`, so the preprocessor carries
**5,846,759 parameters against 369,639** for the 768-d `calm` incumbent it replaced
(`notes/experiments.019-simb-multimodal.expression-strand-retrospective.md:545-560`).

Every strain gets the same token: the encoder runs once, at batch 1, on the wild-type
graph.

### 1.2 The one embedding factor that was ever measured large

The v10 2^4 grid (`results/v10_grid_factorial.json`, 32 runs, 16 cells, 2 seeds per cell,
matched budget 990 epochs, metric = max of a 5-epoch rolling mean of
`val/expression/pearson_per_feature`, an upward-biased order statistic):

| factor | contrast | effect | t |
|---|---|--:|--:|
| embedding | `random_1024` -> `prot_T5_all` | **+0.0682** | +7.84 |
| trunk | L6_h90 -> L2_h45 | -0.0120 | -1.38 |
| readout | mlp -> linear | +0.0072 | +0.83 |
| weight decay | 1e-8 -> 1e-4 | +0.0046 | +0.53 |

Pooled within-cell sd 0.0246 (all 32) or 0.0122 after dropping one collapsed run. This is
the origin of the "content is the only lever" claim, and it is sound as far as it goes:
ProtT5 beats a width-matched random vector by about 0.07. It says nothing about ProtT5
against another real embedding.

### 1.3 The v11 embedding round, read out here for the first time **[NEW]**

v11 ran the four stacks at 1,400 epochs, three model seeds, split seed 0, everything else
pinned. I pulled `val/expression/pearson_per_feature@k0` for all 12 runs and scored
`roll_max` over an 11-epoch window, truncated to epoch 1,395 so the one crashed run is
comparable.

| arm | seed 0 | seed 1 | seed 2 | mean (healthy) |
|---|--:|--:|--:|--:|
| `E_calm` [calm], 768 | **0.017** (collapsed) | 0.175 | 0.162 | 0.168 (n=2) |
| `E_ptt5` [prot_T5_all], 1,024 | 0.148 | 0.149 | 0.130 | 0.143 |
| `E_calm_ptt5`, 1,792 | 0.170 | 0.165 | 0.183 | 0.173 |
| `E_full` (incumbent), 3,328 | 0.174 | 0.158 | 0.179 | 0.170 |

Paired within-seed contrasts (roll_max):

- `E_calm_ptt5 - E_ptt5` = +0.0214, +0.0164, +0.0528; mean **+0.030**, sign 3/3.
- `E_full - E_calm_ptt5` = +0.0042, -0.0076, -0.0035; mean **-0.002**, a clean null.
- `E_ptt5 - E_calm` = +0.132, -0.026, -0.032; the positive seed is the collapsed `E_calm`
  run, so on the two healthy seeds ProtT5 alone is **worse** than CaLM alone by 0.029.

Three findings the campaign has not recorded:

1. **The two cis-regulatory windows buy nothing.** `E_full` adds `fudt_upstream` and
   `fudt_downstream` (1,536 dimensions, 5.5M preprocessor parameters) for -0.002 +/- 0.006.
   The hypothesis in the v11 header, that a promoter embedding pays off on the reporter
   side even though it is at the random floor on the deletion side, is **measured and
   null**.
2. **The incumbent was pinned to `E_full` before v11 read out.** v12's config pins
   `E_full` and the retrospective states the reason plainly:
   "v11 is at epoch ~200 of 1,400, so E_full is unproven against calm"
   (`expression-strand-retrospective.md:654`). Every round since v12 carries 16x the
   preprocessor parameters of `calm` for a measured null.
3. **ProtT5 alone is not the best single embedding on this task.** CaLM alone matches or
   beats it at 1,400 epochs, and the pair beats either.

Run ids and URLs:

E_calm seed 0 (collapsed to 0.017, the failure mode discussed in 1.5)

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/crcbyxju

E_calm seed 1

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/9zgjaaka

E_calm seed 2

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/lpdl0dhm

E_ptt5 seeds 0 / 1 / 2

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/zo00qref

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/2jtwlun9

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/h49944xk

E_calm_ptt5 seeds 0 / 1 / 2

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/zfoypx7v

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/ytgd9r8z

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/xl53ka4y

E_full (the incumbent stack) seeds 0 / 1 / 2

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/t7ovdev5

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/gbj31s5w

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/t8n7rmax

### 1.4 The parameter-free ranking of all 28 representations agrees

`results/baselines_embedding_study.json` (four split seeds each, B3 = neighbor mean, B2 =
rank-swept bilinear ridge; the same partitions the trained arms use). Expression panel,
sorted by B3 validation:

| representation | family | dim | B3 val | sd | B3 test | sd |
|---|---|--:|--:|--:|--:|--:|
| prot_T5 + esm2 | composite | 2,304 | 0.1257 | 0.023 | 0.1073 | 0.038 |
| esm2 650M | protein LM | 1,280 | 0.1222 | 0.023 | 0.0932 | 0.017 |
| prot_T5 + calm | composite | 1,792 | 0.1206 | 0.016 | 0.1082 | 0.047 |
| **prot_T5** | protein LM | 1,024 | **0.1200** | 0.010 | **0.1185** | 0.051 |
| **model_stack (= E_full)** | composite | 3,328 | **0.1009** | 0.014 | **0.0911** | 0.022 |
| calm | coding | 768 | 0.1006 | 0.023 | 0.0998 | 0.015 |
| codon_frequency | coding | 64 | 0.0787 | 0.014 | 0.0620 | 0.011 |
| random_1024 | control | 1,024 | 0.0725 | 0.018 | 0.0179 | 0.016 |
| species_lm 5'+3' | regulatory DNA | 1,536 | 0.0617 | 0.030 | 0.0480 | 0.006 |
| normalized_chrom_pathways | graph | 197 | 0.0329 | 0.021 | 0.0024 | 0.008 |
| every NT window | regulatory DNA | 2,560 | 0.026 to 0.048 | | 0.011 to 0.024 | |

Two things to carry: the incumbent's `model_stack` is a **worse** perturbation
representation than ProtT5 alone on both baselines and both splits; and `random_1024`
reaching 0.0725 on validation against 0.0179 on test is the selection optimism of a
val-chosen k, which is why every claim below quotes test as well.

### 1.5 An optimization failure mode worth naming

Two of 44 runs across v10 and v11 collapse to the chance band and stay there for the whole
budget: `te8272kk` in v10 (`c1_ptt5_big_mlp_wd0`, roll_max 0.0188, flagged in
`v10_grid_factorial.json` as `chance_band_runs`) and `crcbyxju` in v11 (`E_calm` seed 0,
roll_max 0.017, tail-100 mean -0.0005). A 1-in-22 dead-run rate silently corrupts any
3-seed paired contrast; v11's headline `E_ptt5 - E_calm` flips sign depending on whether
that run is included.

https://wandb.ai/zhao-group/torchcell_019_expr_v10/runs/te8272kk

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/crcbyxju

---

## 2. Is the graph used at all?

### 2.1 What the graph is

Nine gene-gene relations, fixed by the genome (6,607 nodes) and not by the Cypher query:

| relation | source | directed in `cell_graph` | edges (excl. self) | genes with >=1 edge |
|---|---|---|--:|--:|
| physical_interaction | SGD `interaction_details`, type Physical | no | 137,604 | 86.6% |
| regulatory_interaction | SGD `regulation_details` | **yes** | 37,703 | 93.0% |
| tflink | TFLink v1.0 | **yes** | 200,643 | 76.8% |
| string12_0_neighborhood | STRING v12 | no | 146,713 | **33.2%** |
| string12_0_fusion | STRING v12 | no | 11,787 | **46.6%** |
| string12_0_cooccurence | STRING v12 | no | 11,085 | **39.4%** |
| string12_0_coexpression | STRING v12 | no | 996,199 | 98.0% |
| string12_0_experimental | STRING v12 | no | 822,094 | 91.1% |
| string12_0_database | STRING v12 | no | 72,617 | 61.0% |

STRING enters at `score > 0` with no confidence cut, and `to_cell_data` drops the edge
weight, so all nine are unweighted topology. Three of the nine have **median undirected
degree 0**: for most genes, heads 3, 4 and 5 at layer 1 can attend only to themselves and
CLS.

### 2.2 How it enters, and how little of the model it touches

- The soft KL regularizer has been **off since `cgt_expr_009`**: `graph_reg_lambda: 0.0`
  resolves for v9 through v15, and all 231 runs across v9-v15 log `graph_reg_lambda = 0`.
- What remains is a **hard boolean attention mask**, one graph per head on heads 0-8, on
  `attention_mask.layers: [1]` of a **six-layer** encoder. Layers 0, 2, 3, 4, 5 are fully
  unconstrained dense attention over 6,608 tokens.
- The mask is **symmetric by construction**:
  `equivariant_cell_graph_transformer.py:2617-2618` writes `head_mask[i,j]` and
  `head_mask[j,i]`, the second line commented `# symmetric`. The two directed TF-target
  relations are symmetrized before the model sees them.
- `perturbation_propagation.enabled: false`. The deletion is **never routed along any
  graph**.

### 2.3 What was measured

**The graph prior is at chance.** `results/graph_prior_probe.json` (1,482 deleted genes x
6,169 reporters): P(a top-1% responder is closer to the deleted gene than a non-responder)
lands in **0.4961 to 0.5057** on all nine graphs, largest excess over the degree-preserving
control **+0.0046**. Longer walks are worse. Direction is the one place signal exists and
the mask destroys it: tflink TF->target **0.5508** (control 0.5017), regulatory_interaction
**0.5239** (control 0.5009).

**[NEW] But the graphs are NOT silent on a different relation, and that relation is the one
this readout needs.** The probe above asked whether deleting X moves X's *neighbors*. A
neighbor-average readout needs something else entirely: do X and Y that are joined on graph
k have *similar deletion profiles*? Measured over the 1,482 single-deletion strains of
fig3_core, 1,097,421 gene pairs, base rate P(profile r > 0.3) = 0.0523, mean pair r =
+0.0070:

| graph | edges within the set | mean r | P(r>0.3) | lift | P(r>0.5) | lift |
|---|--:|--:|--:|--:|--:|--:|
| **string12_0_cooccurence** | 3,923 | **+0.157** | **0.2506** | **4.79x** | 0.0482 | **8.47x** |
| **string12_0_database** | 4,348 | **+0.107** | **0.1874** | **3.59x** | 0.0741 | **13.01x** |
| string12_0_fusion | 964 | +0.024 | 0.0871 | 1.67x | 0.0145 | 2.55x |
| physical_interaction | 16,844 | +0.026 | 0.0863 | 1.65x | 0.0235 | 4.13x |
| string12_0_coexpression | 91,379 | +0.035 | 0.0805 | 1.54x | 0.0139 | 2.45x |
| string12_0_experimental | 105,292 | +0.026 | 0.0720 | 1.38x | 0.0128 | 2.25x |
| string12_0_neighborhood | 4,667 | +0.022 | 0.0711 | 1.36x | 0.0137 | 2.41x |
| **tflink** | 43,526 | +0.009 | 0.0514 | **0.98x** | 0.0060 | 1.05x |
| **regulatory_interaction** | 6,424 | +0.007 | 0.0447 | **0.85x** | 0.0062 | 1.09x |

This inverts the ranking of the earlier probe, and both results are consistent once the two
relations are kept apart:

- **TF -> target predicts which reporters move when you delete the TF** (graph-prior probe,
  tflink 0.5508) **and says nothing about whether two deletions give the same profile**
  (0.98x and 0.85x, at or below chance). Correct biologically: a TF and its target are not
  expected to share a deletion phenotype.
- **Curated pathway/complex co-membership and phylogenetic co-occurrence do the opposite.**
  `string12_0_database` is STRING's curated pathway and complex channel (KEGG, Reactome,
  BioCyc, GO complexes) and `string12_0_cooccurence` is phylogenetic-profile similarity.
  Genes joined on either are 3.6x and 4.8x more likely than base to have correlated
  deletion profiles, and 13.0x and 8.5x at the r > 0.5 threshold. This is the "same complex
  implies same deletion phenotype" relation, and it is measurably present.

Two things follow immediately. First, **the complex/pathway relation that section 3 asks
about is already in the graph, as the `string12_0_database` channel**, and it is the
strongest single relation on this statistic after co-occurrence. Second, those two channels
are exactly the ones the coverage table flags as near-silent: co-occurrence touches 39.4%
of genes and database 61.0%, both with low median degree, so at layer 1 of the mask the two
informative heads are dark for most genes while the two at-chance TF heads are dense. The
mask spends its head budget in inverse proportion to the measured signal.

Hypothesis (untested, and the graph-as-retrieval-key measurement in section 4(d) is its
test): these two relations are useful as a **retrieval prior on the readout**, not as a
sparsity mask on one attention layer, because what they encode is "these two deletions land
in the same place", which is a statement about the output and not about where a token may
look.

**The `graphreg_off` arm was never run.** `conf/gh_cgt_multitask_sweep_graphreg_off_005.yaml`
exists and sets `graph_reg_lambda: 0.0`; zero runs in
`torchcell_019-simb-multimodal_cgt_multitask` (496 runs enumerated) carry the `sweep` or
`graphreg_off` tag, and zero rows of `round_leaderboards.csv` (2,309 rows) match.

**No mask-off arm has ever been run in the expression campaign either.** Every run in v9
(81), v10 (68), v11 (12), v12 (42), v13 (24) and v15 (4) has `graph_prior = "mask"` and
`n_masked_heads = 9`. The `P_free` arm the graph-prior probe recommended was never
launched.

What *was* measured about the graph channel:

- **v7 Optuna marginal, lambda = 0 vs lambda > 0** (289 trials, Halton-sampled, not
  paired, metric = max of a 5-epoch rolling mean, median 65-70 epochs): mean 0.0356
  (n=141) vs 0.0507 (n=148), Welch t = 3.44, p = 6.7e-4. But the **best** trial in each
  arm is 0.1389 vs 0.1631, a gap of 0.024 against the replicate sd of 0.0222. The penalty
  raises the average trial and is not distinguishable at the best trial.

https://wandb.ai/zhao-group/torchcell_019_expr_v7/runs/nrurev80

https://wandb.ai/zhao-group/torchcell_019_expr_v7/runs/lcjezzfr

- **KL vs hard mask, n = 1 per arm** (v8 wave 1): `A0_baseline` 0.1575, `D2_mask` 0.1494,
  delta -0.008 inside the noise floor. The mask was frozen in from `cgt_expr_009` onward on
  a **throughput** argument (fused SDPA, ~1.6x epochs per unit wall clock), not an accuracy
  win.

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/w5ob5tku

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/au8fu60o

- Replicate spread of the incumbent configuration, n = 8 at 9,900 epochs:
  **0.1965 +/- 0.0222** (`results/launch_plan_evidence.json`).

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/hx8pxdic

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/u1vuznme

The incumbent split round, for reference:

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/wq8y8nd5

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/bn37i9vs

### 2.4 So: is it a set transformer with a perturbation token?

**Yes, and the model file says so itself.** From the `PerturbationGraphPropagation`
docstring (`torchcell/models/equivariant_cell_graph_transformer.py:740-770`), with the
numerical verification in the same docstring:

> the encoder runs at batch 1 on the WILDTYPE graph, so `H_genes` and `h_CLS` are
> identical for every strain; the only strain-dependent step is
> `EquivariantPerturbationTransform`, whose K/V set is just the |S_b| perturbed tokens.
> For a SINGLE deletion the softmax is over one key, so the attention weight is exactly 1
> for every query gene and the attended vector is query-INDEPENDENT ... Verified
> numerically (max |attended_i - attended_0| = 0). ... The nine graphs shape h_i but never
> CARRY the deletion.

So for ~96% of the training rows the model is exactly

  yhat(strain b, reporter i) = R(h_i, c_b),  c_b = W_O W_V h_{p(b)} in R^90,

with `h_i` the same for every strain. It is a permutation-equivariant set encoder over
frozen sequence embeddings, with one strain-level 90-d vector added to every token. The
graph appears in one of six attention layers, symmetrized, on a target measured to be at
chance, with the KL off, on a module that never carries the deletion. **Calling the
incumbent a set transformer over gene embeddings with a perturbation token is accurate,
not a simplification.**

That structure also puts the ceiling in the same place the low-rank analysis does: the
whole 6,169-gene prediction is a function of one 90-d vector that is itself a linear
function of the deleted gene's frozen embedding.

---

## 3. TF-target, complexes and pathways: what is there and what is absent

| mechanistically relevant relation | present? | where |
|---|---|---|
| TF -> target (curated) | **yes**, `regulatory_interaction` | SGD `regulation_details`, DiGraph, 37,703 directed edges, only 52 reciprocal |
| TF -> target (compendium) | **yes**, `tflink` | TFLink v1.0, DiGraph, 200,643 directed edges, max degree 4,908 |
| YEASTRACT specifically | **no** | not ingested anywhere in torchcell |
| protein complexes | **not as a named relation, but the proxy is present and is the best channel there is** | no CYC2008, no EBI Complex Portal, no `protein_complex` builder. `locus.py:324` has a passive `complexes: list[str]` field nothing reads. `string12_0_database` folds curated complex and pathway co-membership into one score, and section 2.3 measures it at 3.6x / 13.0x lift on deletion-profile similarity |
| GO co-annotation as a gene-gene edge | **no** | `G_go` is the ontology DAG, deliberately excluded from `SCEREVISIAE_GENE_GRAPH_MAP`; GO can only enter as an incidence graph, and v13 passes `incidence_graphs=None` |
| metabolic / pathway membership in fig3_core | **no** | `YeastGEM().bipartite_graph` is attached only when `per_metabolite` is an active head; v13's `active_heads` is `['per_gene']`. Verified: the built `cell_graph` carries exactly nine gene-gene edge types and no metabolite or reaction node type |
| co-expression from the training compendium (GEARS-style) | **no** | not built. `string12_0_coexpression` is STRING's precomputed cross-compendium channel, not this panel's |

**Is there a regulatory edge type? Yes, two, and the model throws away the only part of
them that carries signal.** The one measurement that beats chance anywhere in the graph
channel is tflink in the TF->target orientation, AUC 0.5508 against a degree control of
0.5017. `_build_head_mask` symmetrizes it back to 0.5057.

### 3.1 Adding a TF-target edge type

Nothing needs to be added. Both directed graphs are already in `cell_graph` with direction
intact; only `_build_head_mask` destroys it. The change is three lines: stop writing
`head_mask[j,i]` for relations declared directed, and add a second head per directed
relation for the reverse orientation so target->TF stays reachable. Cost: a config flag
plus a masked-vs-directed paired arm. This is the cheapest evidence-backed change on the
whole input side.

YEASTRACT would add a third, independently curated TF-target set; given that TFLink
already sits at 0.55 in the informative orientation and that TFLink is itself a harmonized
union of such resources, I would run the directed-mask arm first and only then ask whether
a different TF-target source moves it.

### 3.2 Adding a co-expression graph from the training compendium

This is available today and leakage-free, and the campaign has explicitly ruled out the
wrong version of it. `expression_baselines.py`'s docstring says:

> each gene is deleted exactly once ... A perturbation representation derived from the
> deleted gene's own observed response (the obvious in-data choice) is therefore undefined
> for every val strain, and was measured to cover 0 of 155.

That is true for the gene's **row** (its own deletion profile). It is **not** true for the
gene's **column**: gene X appears as a *reporter* in every training strain, so
`R_train[:, X]` is a 1,186-dimensional representation of X that uses no validation row and
is defined for 90.3% of validation strains (140/155 on seed 0, the rest being doubles or
genes absent from the reporter set). The relation it encodes is exactly the
mechanistically relevant one: if deleting Y moves X, then X and Y are likely in the same
process, so deleting X probably moves what deleting Y moved.

**[NEW] I built it and measured it on all four v13 split seeds** (neighbor average, k swept
on validation, per-feature Pearson identical to the training metric):

| representation | dim | val mean | sd | test mean | sd |
|---|--:|--:|--:|--:|--:|
| `oracle_own_profile` (see section 4) | 6,169 | **0.6289** | 0.011 | **0.6455** | 0.011 |
| prot_T5 | 1,024 | 0.1073 | 0.019 | 0.1157 | 0.057 |
| prot_T5 + calm | 1,792 | 0.1073 | 0.024 | 0.1014 | 0.058 |
| prot_T5 + coexpr_svd32 | 1,056 | 0.1011 | 0.030 | 0.0971 | 0.019 |
| prot_T5 + coexpr_svd8 | 1,032 | 0.0986 | 0.017 | 0.0892 | 0.012 |
| calm | 768 | 0.0844 | 0.014 | 0.0947 | 0.025 |
| calm + coexpr_svd8 | 776 | 0.0887 | 0.012 | 0.0837 | 0.017 |
| **coexpr_col** (train-split co-expression) | 1,186 | **0.0710** | 0.008 | **0.0754** | 0.022 |
| coexpr_svd8 | 8 | 0.0644 | 0.004 | 0.0491 | 0.007 |
| coexpr_svd32 | 32 | 0.0546 | 0.016 | 0.0527 | 0.019 |
| random, matched width (seed 0 only) | 1,186 | -0.0057 | | +0.0072 | |

Paired across the four splits, the fusion contrasts are null to slightly negative:
`prot_T5+coexpr_svd32 - prot_T5` = -0.006 +/- 0.017 on validation and -0.019 +/- 0.047 on
test; `calm+coexpr_svd8 - calm` = +0.004 +/- 0.006 on validation, -0.011 +/- 0.030 on test.

**The honest reading: the train-split co-expression representation carries real signal
(0.071 to 0.075 against a matched-width random floor of about 0.00) but is NOT
complementary to a protein language model under neighbor averaging.** The GEARS-style
co-expression input, in its naive form, does not close the gap. That is a measured null and
it should stop the obvious arm from being run at full cost.

It is not a null for everything. **[NEW]** In a ridge from the gene representation to the
top-32 coefficients of the train gene basis (split seed 0), adding the co-expression column
to ProtT5 raises the rank-8 reconstruction from **0.140 val / 0.079 test to 0.153 val /
0.092 test**, and the per-component validation r on PC2 from +0.37 to +0.47, on PC7 from
+0.18 to +0.25, on PC8 from +0.26 to +0.28. n = 1 split and the ridge penalty is chosen on
validation, so this is suggestive, not established.

---

## 4. The oracle test: how far is the best possible embedding from the current one? **[NEW]**

The neighbor rule fixes the function class; the only thing that varies is the retrieval
key. Swapping the key for the strain's own measured profile gives the ceiling of that
function class, i.e. what a *perfect* gene embedding would buy if the embedding's only job
were to find the right training strains.

| retrieval key | val | test |
|---|--:|--:|
| the strain's own measured profile (oracle) | **0.629 +/- 0.011** | **0.646 +/- 0.011** |
| ProtT5, the best real embedding | 0.107 +/- 0.019 | 0.116 +/- 0.057 |
| the trained CGT (n=8, 9,900 epochs) | 0.1965 +/- 0.0222 | -- |
| replicate reproducibility ceiling of the label | 0.775 | |

**The gap between the best possible embedding and the current one is about 0.52 Pearson,
and the trained model closes about a fifth of it.** Two caveats stated plainly: the oracle
uses the held-out strain's own label, so it is a bound and not an achievable score; and k
is chosen on validation for every row, which is why the test column is quoted alongside
(here the two agree to 0.02, so the selection optimism is small at this n).

Three follow-ups that locate the gap.

**(a) The oracle's advantage is not concentrated in the loud deletions.** Stratifying the
seed-0 validation strains by the norm of their response profile, at k = 10:

| retrieval key | Q1 (quietest) | Q2 | Q3 | Q4 (loudest) | all |
|---|--:|--:|--:|--:|--:|
| oracle | +0.551 | +0.579 | +0.609 | +0.646 | +0.625 |
| prot_T5 | +0.010 | +0.076 | +0.088 | +0.126 | +0.090 |
| calm | +0.036 | +0.060 | +0.098 | +0.073 | +0.070 |
| coexpr_col | +0.016 | +0.042 | +0.016 | +0.079 | +0.045 |
| normalized_chrom_pathways | +0.009 | -0.001 | +0.033 | +0.029 | +0.021 |
| mean profile norm | 8.59 | 10.23 | 13.15 | 23.62 | |

The oracle works about as well on the quietest quartile as on the loudest. Sequence
embeddings only work on the loudest deletions and are at zero on the quietest. The failure
is not "most deletions do nothing"; the structure is there in every quartile and the
embedding cannot see it.

**(b) The geometry, measured directly.** Over the 703,230 pairs of deleted genes in the
seed-0 training rows, the Spearman between embedding cosine similarity and deletion-profile
correlation:

| representation | rho |
|---|--:|
| coexpr_col | +0.0182 |
| prot_T5 | +0.0171 |
| calm | +0.0141 |
| normalized_chrom_pathways | +0.0116 |

against a profile-correlation distribution with mean +0.009, 99th percentile +0.457,
maximum +0.938, and 5.48% of pairs above +0.3. This is the cleanest statement of the
problem I can make: **the co-response structure is real and none of our gene
representations is aligned with it.**

**(c) Only two or three directions of the response are predictable at all.** Ridge from the
deleted gene's representation to the coefficients on the train gene basis, split seed 0,
per-component validation r:

| representation | PC1 | PC2 | PC3 | PC4 | PC7 | PC8 | rank-8 recon val | test |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| prot_T5 | +0.33 | +0.37 | +0.24 | +0.03 | +0.18 | +0.26 | 0.140 | 0.079 |
| calm | +0.16 | +0.42 | +0.19 | +0.01 | +0.08 | +0.18 | 0.104 | 0.077 |
| prot_T5 + coexpr_col | +0.29 | +0.47 | +0.21 | +0.07 | +0.25 | +0.28 | 0.153 | 0.092 |
| coexpr_col | +0.15 | +0.27 | +0.03 | -0.01 | +0.22 | +0.10 | 0.082 | 0.073 |

For reference, the rank-r reconstruction ceilings with coefficients known
(`results/lowrank_output_ceiling.json`): rank 1 = 0.266, rank 2 = 0.412, rank 4 = 0.525,
rank 8 = 0.579, rank 16 = 0.666, rank 32 = 0.727. So a *perfect* rank-2 placement already
beats the trained model (0.412 vs 0.197), and the genotype currently resolves PC1 and PC2
at r ~ 0.3 to 0.45 and essentially nothing beyond PC3.

---

## 5. Would an expression-derived gene embedding close the gap?

**Measured, on the one version available today: no.** Section 3.2's `coexpr_col` is exactly
the leakage-free SVD-of-the-train-compendium embedding, and it reaches 0.071/0.075 with a
null fusion against ProtT5. The evidence in the repo points the same way in every other
form:

- `normalized_chrom_pathways`, the one graph-derived vector already in the builder, is the
  **best** representation on morphology in the kNN probe (0.100, above every sequence
  embedding) and near the floor on expression (B3 val 0.033, test 0.002). Functional
  annotation transfers to morphology and not to the transcriptome.
- `string12_0_coexpression` is a cross-compendium co-expression graph and is one of the
  nine at chance in the graph-prior probe (0.5008 against a 0.4982 degree control), and
  falls below 0.5 at t = 3.
- The published benchmark position is the same: Ahlmann-Eltze, Huber and Anders 2025
  (doi:10.1038/s41592-025-02772-6) found no deep model beating simple baselines, and their
  linear model beat GEARS **while using GEARS's own perturbation embeddings**. GEARS's
  inputs are a GO similarity graph and a co-expression graph; the benchmark's finding is
  that those inputs do not carry more than a bilinear map can use.

What has **not** been tested, and is the honest open case:

- **A pretrained single-cell or compendium foundation-model embedding** (Geneformer,
  scGPT, or a yeast-specific equivalent trained on SPELL). None is in the builder. SPELL is
  documented in `notes/torchcell.datasets.scerevisiae.spell.md` with a working PCL loader,
  but **it is not built on this machine** (no `data/spell` under DATA_ROOT). Hypothesis
  (untested): an embedding trained across ~200 yeast studies encodes co-regulation that
  neither a protein LM nor this single panel's 1,186 columns can.
- **A functional-profile embedding from a different assay.** The strongest candidates are
  already built LMDBs: the SGA genetic-interaction profile
  (`dmi_costanzo2016`, ~4,000-dimensional per query gene) and the chemical-genomic fitness
  profiles (`env_chemgen_hillenmeyer2008`, `env_chemgen_hoepfner2014`, and six others).
  Hypothesis (untested): the SGA profile is the canonical functional gene embedding in
  yeast and should align with deletion-response similarity far better than sequence does,
  for the same reason SGA profiles cluster genes into pathways. This is the single highest
  expected-value untested input in the repo, and section 4(b)'s rho is the exact
  cheap gate for it.

---

## 6. Proposed input-side arms, with cost and hypothesis

Ordered by (measured evidence behind it) / (cost). Each is stated as a hypothesis where it
is one.

### Tier 0: CPU probes, hours, no GPU

**A1. Retrieval-alignment gate for every candidate representation.** Extend section 4(b)
to the SGA interaction profile (`dmi_costanzo2016`), each chemical-genomic fitness profile,
GO semantic similarity, and any new embedding, and report rho against the +0.017 ProtT5
baseline and the +0.457 p99 of the profile-correlation distribution. **Cost: under a day of
CPU.** Hypothesis (untested): the SGA profile reaches rho > 0.05, which would be a 3x
improvement on anything measured and would justify a GPU arm; anything at rho < 0.02 should
not get a GPU arm at all. **This gate should run before any of the tiers below.** It is the
cheapest thing in this report and it is the one measurement that decides whether the input
side is worth more GPU time.

**A2. Recompute the kNN oracle on the proteome panel.** The same script, `fig3_proteome`,
`protein_abundance`. **Cost: an hour.** Hypothesis (untested): the proteome oracle sits far
below 0.63 because the label ceiling is 0.42 to 0.61, which would say the proteome round's
headroom is small and reprioritize the queued proteome work.

### Tier 1: cheap GPU arms, evidence already in hand

**B1. Directed attention mask.** Stop symmetrizing `regulatory_interaction` and `tflink`;
give each a forward head and a reverse head. **Cost: a config flag plus ~10 lines in
`_build_head_mask`; one paired arm, 3 seeds x 2 arms at 1,400 epochs, about 6 Delta A40
runs, ~300 GPU-hours.** Evidence: TF->target AUC 0.5508 vs symmetric 0.5057 vs control
0.5017 (`graph_prior_probe.json`). Hypothesis (untested): the gain is small, because a
+0.05 AUC prior applied at one of six layers is a weak constraint, but it is the only
graph-channel change with a positive data-side measurement behind it, and it is nearly
free.

**B2. Mask-off (`P_free`) control, finally run.** Nine heads at layer 1 unconstrained,
everything else pinned. **Cost: 3 seeds, 1,400 epochs, 3 runs.** Evidence: the mask's
target is at chance on all nine graphs; the mask has never been ablated in the expression
campaign; it was adopted on a throughput argument with an n = 1 accuracy comparison inside
the noise. Hypothesis (untested): null within +/- 0.02, which would let the paper state
plainly that the graph channel is inert for this task rather than leaving it implied.
**This arm is worth running even if it is null**, because every graph claim in the
expression document currently rests on an unablated component.

**B3. Retire `E_full` to `E_calm_ptt5`.** Drop `fudt_upstream` and `fudt_downstream`.
**Cost: zero, it is a config edit, and it makes every subsequent run cheaper** (3,328 ->
1,792 input, 5.85M -> 1.8M preprocessor parameters). Evidence: `E_full - E_calm_ptt5` =
-0.002 +/- 0.006 paired over 3 seeds; `model_stack` B3 test 0.091 vs `prot_T5` 0.119 over 4
splits. This is not an arm, it is a correction.

### Tier 2: the input arms that actually target the measured gap

**C1. SGA genetic-interaction profile as the perturbation representation.** Concatenate the
deleted gene's Costanzo/Kuzmin interaction-profile vector (already an LMDB) to the sequence
token. **Cost: a new `NodeEmbeddingBuilder` entry plus a coverage audit (the SGA query set
does not cover all 6,607 genes, so a missing-value policy is required and must be declared,
not imputed silently); then 3 seeds x 2 arms at 1,400 epochs.** Gated on A1. Hypothesis
(untested): this is the representation most likely to align with deletion-response
similarity, because both quantities are "what pathway is this gene in" read out through a
deletion.

**C2. Reporter-side co-expression token.** Section 3.2 used the co-expression column as the
*perturbation* representation. The same vector is also available for every reporter, and the
reporter side has never been tested (`graph-prior-probe.md:157-159` lists it as open
hypothesis 2). Give each gene token the concatenation `[sequence embedding ; its own
train-split response column]`, which tells the model both what the gene is and how it tends
to respond. **Cost: an embedding-builder entry that reads the train split, which means the
embedding is split-dependent and must be rebuilt per split seed; 3 seeds x 2 arms.**
Hypothesis (untested): this helps more than the perturbation-side version did, because the
per-gene head currently learns responsiveness from scratch for every reporter while the
information is sitting in the training matrix.

**C3. Yeast expression foundation embedding from SPELL.** Build the SPELL compendium (the
loader exists), fit an embedding (SVD or a masked autoencoder) over ~200 studies with
Kemmeren and Sameith excluded, and use it as a gene representation. **Cost: dataset build
plus embedding training, on the order of a week of engineering before any GPU arm; the
arm itself is then 3 seeds x 2.** Hypothesis (untested): an external compendium carries
co-regulation structure this panel's 1,186 columns cannot, which is the one form of the
expression-derived embedding that section 5's null does not cover.

### Tier 3: structural, and larger than an input change

**D1. Put the deletion inside the encoder.** The measured additivity
(`equivariant_cell_graph_transformer.py:740-770`, max |attended_i - attended_0| = 0) means
no readout or objective change can enlarge the hypothesis class while the perturbation is a
single post-encoder key. This is already on the queue in `3-next.tex`; section 4(b) is a
second, independent argument for it: even a perfect pair term cannot help if the retrieval
geometry is absent, but an absent pair term guarantees it cannot.

**D2. Learn the retrieval metric instead of the profile.** Train a metric head on
`(gene_i, gene_j) -> profile correlation` over training pairs, on top of whatever
representations A1 admits, and use the learned similarity for a neighbor average.
**Cost: CPU or one GPU-day; it is a much smaller learning problem than the full profile.**
Hypothesis (untested): the target has 703k labeled pairs against 1,186 training profiles,
so it is a far better-posed problem than the current one, and section 4 says its ceiling is
0.63 rather than 0.20.

---

## 7. What I would not do

- Do not build the `P_graph` arm (a pair term as a function of network distance): the
  network does not carry that relationship (`graph_prior_probe.md`).
- Do not spend a round on more nucleotide or promoter representations. Six NT windows and
  two species-LM flanks are all at or below the random floor on both baselines and on the
  neighbor probe, and the one round that tested them inside the model (v11 `E_full`)
  measured -0.002.
- Do not run a full-cost GPU round on the naive GEARS-style co-expression input. Section
  3.2 measured it.

---

## Scripts I ran for the [NEW] numbers

All read-only, in the session scratchpad, not written into the repo. If any of these
numbers is to be used in the document or the paper, the script must be moved into
`experiments/019-simb-multimodal/scripts/` per the repo's artifact rule.

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/oracle_embed.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/oracle_embed2.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/pc_predictability.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/retrieval_structure.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/edge_profile_similarity.py
