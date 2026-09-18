# Agent 08: the inputs. What does the model see about each gene and about the graph?

2026-09-18 review round. Read-only. Every number traces to a file, a run id, or a script I
ran in the session scratchpad. New measurements made in this session are marked **[NEW]**;
their scripts are scratchpad scripts, not committed artifacts, so they are evidence for a
decision and would need to move into `experiments/019-simb-multimodal/scripts/` before any
of these numbers enters a note or the paper.

---

## 0. The headline, which is not the one I expected to write

**[NEW] A parameter-free neighbor average that retrieves on the interaction-graph adjacency
row matches the trained transformer.** Per-feature Pearson on the v13 partitions, all four
split seeds, k chosen on validation:

| retrieval key | val | test |
|---|--:|--:|
| the strain's own measured profile (oracle) | 0.6293 +/- 0.0103 | 0.6455 +/- 0.0112 |
| **`string12_0_experimental` adjacency row** | **0.1964 +/- 0.0160** | 0.1745 +/- 0.0302 |
| **union of all nine graphs, adjacency row** | **0.1889 +/- 0.0203** | **0.2013 +/- 0.0195** |
| `string12_0_coexpression` adjacency row | 0.1877 +/- 0.0161 | 0.1809 +/- 0.0207 |
| `string12_0_database` adjacency row | 0.1606 +/- 0.0188 | 0.1764 +/- 0.0113 |
| `physical_interaction` adjacency row | 0.1485 +/- 0.0145 | 0.1895 +/- 0.0201 |
| ProtT5, the best sequence embedding | 0.1052 +/- 0.0183 | 0.1269 +/- 0.0334 |
| `tflink` adjacency row | 0.0408 +/- 0.0125 | 0.0104 +/- 0.0225 |
| `regulatory_interaction` adjacency row | 0.0375 +/- 0.0276 | 0.0362 +/- 0.0156 |
| **the trained CGT** (n = 8, 9,900 epochs, max over epochs) | **0.1965 +/- 0.0222** | -- |
| replicate reproducibility ceiling of the label | 0.775 | |

Paired within split seed against ProtT5, every one of the six graph keys is positive on
**4 of 4 seeds** on both validation and test. Against each graph's own degree-preserving
configuration-model rewiring: `string12_0_experimental` **+0.182 +/- 0.016**,
`string12_0_coexpression` +0.170 +/- 0.024, `string12_0_database` +0.166 +/- 0.029,
`physical` +0.126 +/- 0.019, and for the two TF-target graphs only +0.024 and +0.015. This
is topology, not a hub artifact.

The model is handed these exact nine graphs. It consumes them as a **symmetric boolean
attention mask on one of six encoder layers**, with the KL regularizer at lambda 0, on a
module that never carries the deletion. It never uses them as a retrieval key. **The single
largest measured source of signal on this task is an input the model already has and
throws away in the form that matters.**

Three supporting facts from the same session:

- The **oracle** for the neighbor-average function class is 0.629 val / 0.646 test. Every
  *node feature* we have reaches 0.10 to 0.13. The gap is about 0.52 Pearson.
- **No gene node feature's geometry aligns with deletion-response similarity.** Over
  703,230 training gene pairs, the Spearman between embedding cosine and profile
  correlation is +0.017 (ProtT5), +0.014 (CaLM), +0.012 (chromatin pathways), +0.018
  (train-split co-expression), against a distribution whose 99th percentile is +0.457 and
  in which 5.5% of pairs exceed +0.3.
- The incumbent's 3,328-dimensional four-embedding stack is **measurably worse** than
  ProtT5 alone as a perturbation representation, and its two cis-regulatory blocks are a
  measured null inside the model.

---

## 1. What is in the token, with dimensions and measured marginal value

### 1.1 The resolved input of the incumbent

`cell_dataset.node_embeddings` resolves for `cgt_expr_v13_split` (and v12, v15, and v11's
`E_full` arm) to:

| slot | name | dim | model |
|---|---|--:|---|
| 5' flank | `fudt_upstream` | 768 | species-aware fungal up/down transformer |
| ORF | `calm` | 768 | codon language model over the ORF |
| protein | `prot_T5_all` | 1,024 | ProtT5 XL UniRef50 |
| 3' flank | `fudt_downstream` | 768 | same fungal model, 3' window |
| | **total** | **3,328** | |

Set in `conf/cgt_expr_v11_emb.yaml` and inherited unchanged. `learnable_embedding.enabled:
false`, `gene_num: 6607`, `hidden_channels: 90`. The input projection's hidden layer is
`(input + 90) / 2`, so the preprocessor carries **5,846,759 parameters against 369,639**
for the 768-d `calm` it replaced. Every strain gets the same token: the encoder runs once,
at batch 1, on the wild-type graph.

### 1.2 The one embedding factor ever measured large

v10's 2^4 grid (`results/v10_grid_factorial.json`; 32 runs, 16 cells, 2 seeds per cell,
matched budget 990 epochs; metric = max of a 5-epoch rolling mean, an upward-biased order
statistic):

| factor | contrast | effect | t |
|---|---|--:|--:|
| embedding | `random_1024` -> `prot_T5_all` | **+0.0682** | +7.84 |
| trunk | L6_h90 -> L2_h45 | -0.0120 | -1.38 |
| readout | mlp -> linear | +0.0072 | +0.83 |
| weight decay | 1e-8 -> 1e-4 | +0.0046 | +0.53 |

Pooled within-cell sd 0.0246 (all 32), 0.0122 after dropping one collapsed run. Sound as
far as it goes: ProtT5 beats a width-matched random vector by ~0.07. It says nothing about
ProtT5 against another *real* embedding.

### 1.3 The v11 embedding round, read out here for the first time **[NEW]**

v11 ran four stacks at 1,400 epochs, three model seeds, split seed 0, everything else
pinned. I pulled `val/expression/pearson_per_feature@k0` for all 12 runs and scored
`roll_max` over an 11-epoch window, truncated to epoch 1,395 so the one crashed run is
comparable.

| arm | seed 0 | seed 1 | seed 2 | mean (healthy) |
|---|--:|--:|--:|--:|
| `E_calm` [calm], 768 | **0.017** (collapsed) | 0.175 | 0.162 | 0.168 (n = 2) |
| `E_ptt5` [prot_T5_all], 1,024 | 0.148 | 0.149 | 0.130 | 0.143 |
| `E_calm_ptt5`, 1,792 | 0.170 | 0.165 | 0.183 | 0.173 |
| `E_full` (the incumbent), 3,328 | 0.174 | 0.158 | 0.179 | 0.170 |

Paired within-seed contrasts:

- `E_calm_ptt5 - E_ptt5` = +0.0214, +0.0164, +0.0528; mean **+0.030**, sign 3/3.
- `E_full - E_calm_ptt5` = +0.0042, -0.0076, -0.0035; mean **-0.002**, a clean null.
- `E_ptt5 - E_calm` = +0.132, -0.026, -0.032; the one positive seed is the collapsed
  `E_calm` run, so on the two healthy seeds ProtT5 alone is **worse** than CaLM alone by
  0.029.

Three findings the campaign has not recorded:

1. **The two cis-regulatory windows buy nothing.** `E_full` adds 1,536 dimensions and 5.5M
   preprocessor parameters for -0.002 +/- 0.006. The v11 header's hypothesis, that a
   promoter embedding pays off on the reporter side even though it is at the random floor
   on the deletion side, is **measured and null**.
2. **The incumbent was pinned to `E_full` before v11 read out.** The retrospective states
   it: "v11 is at epoch ~200 of 1,400, so E_full is unproven against calm"
   (`expression-strand-retrospective.md:654`). Every round since v12 has carried 16x the
   preprocessor parameters of `calm` for a measured null.
3. **ProtT5 alone is not the best single embedding on this task.** CaLM matches or beats it
   at 1,400 epochs; the pair beats either.

E_calm seeds 0 / 1 / 2 (seed 0 is the collapsed run of section 1.5)

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/crcbyxju

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/9zgjaaka

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/lpdl0dhm

E_ptt5 seeds 0 / 1 / 2

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/zo00qref

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/2jtwlun9

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/h49944xk

E_calm_ptt5 seeds 0 / 1 / 2

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/zfoypx7v

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/ytgd9r8z

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/xl53ka4y

E_full, the incumbent stack, seeds 0 / 1 / 2

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/t7ovdev5

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/gbj31s5w

https://wandb.ai/zhao-group/torchcell_019_expr_v11/runs/t8n7rmax

### 1.4 The parameter-free ranking of all 28 representations agrees

`results/baselines_embedding_study.json`, four split seeds each, B3 = neighbor mean,
B2 = rank-swept bilinear ridge, on the partitions the trained arms use. Expression panel,
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
| every NT window | regulatory DNA | 2,560 | 0.026-0.048 | | 0.011-0.024 | |

Carry two things. The incumbent's `model_stack` is a **worse** perturbation representation
than ProtT5 alone on both baselines and both splits. And `random_1024` reaching 0.0725 on
validation against 0.0179 on test is the selection optimism of a val-chosen k, which is why
every claim in this report quotes test as well.

### 1.5 An optimization failure mode worth naming

Two of 44 runs across v10 and v11 collapse to the chance band and stay there for the whole
budget: `te8272kk` in v10 (roll_max 0.0188, flagged as `chance_band_runs` in
`v10_grid_factorial.json`) and `crcbyxju` in v11 (`E_calm` seed 0, roll_max 0.017, tail-100
mean -0.0005). A 1-in-22 dead-run rate silently corrupts any 3-seed paired contrast; v11's
`E_ptt5 - E_calm` flips sign depending on whether that run is included.

https://wandb.ai/zhao-group/torchcell_019_expr_v10/runs/te8272kk

---

## 2. Is the graph used at all?

### 2.1 What the graph is

Nine gene-gene relations over 6,607 nodes fixed by the genome, not by the Cypher query:

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
degree 0**.

### 2.2 How it enters, and how little of the model it touches

- The soft KL regularizer has been **off since `cgt_expr_009`**: `graph_reg_lambda: 0.0`
  resolves for v9 through v15, and all 231 runs across those projects log it as 0.
- What remains is a **hard boolean attention mask**, one graph per head on heads 0-8, on
  `attention_mask.layers: [1]` of a **six-layer** encoder. Layers 0, 2, 3, 4, 5 are fully
  unconstrained dense attention over 6,608 tokens.
- The mask is **symmetric by construction**: `equivariant_cell_graph_transformer.py:2617-2618`
  writes `head_mask[i,j]` and `head_mask[j,i]`, the second commented `# symmetric`. The two
  directed TF-target relations are symmetrized before the model sees them.
- `perturbation_propagation.enabled: false`. The deletion is **never routed along any
  graph**.
- A gene with no edge in a relation can attend, on that head at layer 1, only to itself and
  CLS. Given the coverage column, that is most genes on heads 3, 4 and 5.

### 2.3 What was measured before this session

**The graph prior is at chance.** `results/graph_prior_probe.json`, 1,482 deleted genes x
6,169 reporters: P(a top-1% responder is closer to the deleted gene than a non-responder)
lands in **0.4961 to 0.5057** on all nine graphs; largest excess over the degree-preserving
control **+0.0046**; longer walks are worse. Direction is the one place signal exists and
the mask destroys it: tflink TF->target **0.5508** (control 0.5017), regulatory_interaction
**0.5239** (control 0.5009).

**The `graphreg_off` arm was never run.** `conf/gh_cgt_multitask_sweep_graphreg_off_005.yaml`
exists; zero of 496 runs in `torchcell_019-simb-multimodal_cgt_multitask` carry the `sweep`
or `graphreg_off` tag, and zero of 2,309 rows of `round_leaderboards.csv` match.

**No mask-off arm has ever been run either.** Every run in v9 (81), v10 (68), v11 (12), v12
(42), v13 (24) and v15 (4) has `graph_prior = "mask"` and `n_masked_heads = 9`. The
`P_free` arm the graph-prior probe recommended was never launched.

What *was* measured about the graph channel:

- **v7 Optuna marginal, lambda = 0 vs lambda > 0** (289 trials, Halton-sampled, not paired,
  max of a 5-epoch rolling mean, median 65-70 epochs): mean 0.0356 (n = 141) vs 0.0507
  (n = 148), Welch t = 3.44, p = 6.7e-4. But the **best** trial in each arm is 0.1389 vs
  0.1631, a gap of 0.024 against a replicate sd of 0.0222. The penalty raises the average
  trial and is not distinguishable at the best trial.

https://wandb.ai/zhao-group/torchcell_019_expr_v7/runs/nrurev80

https://wandb.ai/zhao-group/torchcell_019_expr_v7/runs/lcjezzfr

- **KL vs hard mask, n = 1 per arm** (v8 wave 1): `A0_baseline` 0.1575, `D2_mask` 0.1494,
  delta -0.008, inside the noise floor. The mask was frozen in from `cgt_expr_009` onward
  on a **throughput** argument (fused SDPA, ~1.6x epochs per unit wall clock), not an
  accuracy win.

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/w5ob5tku

https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/au8fu60o

- Replicate spread of the incumbent, n = 8 at 9,900 epochs: **0.1965 +/- 0.0222**
  (`results/launch_plan_evidence.json`).

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/hx8pxdic

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/u1vuznme

The incumbent split round, for reference:

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/wq8y8nd5

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/bn37i9vs

### 2.4 **[NEW]** The graphs are not silent. They were asked the wrong question.

The graph-prior probe asked whether deleting X moves X's *neighbors*. A neighbor-average
readout needs a different relation: do X and Y that are joined on graph k have *similar
deletion profiles*? Measured over the 1,482 single-deletion strains of fig3_core,
1,097,421 gene pairs, base rate P(profile r > 0.3) = 0.0523, mean pair r = +0.0070:

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

Both probes are right, and they measure different relations:

- **TF -> target predicts which reporters move when you delete the TF** (0.5508) and says
  **nothing** about whether two deletions give the same profile (0.98x, 0.85x). Correct
  biologically: a TF and its target are not expected to share a deletion phenotype.
- **Curated pathway and complex co-membership (`string12_0_database`: KEGG, Reactome,
  BioCyc, GO complexes) and phylogenetic co-occurrence do the opposite**, at 3.6x and 4.8x
  the base rate and 13.0x and 8.5x at r > 0.5. This is the "same complex implies same
  deletion phenotype" relation, and it is measurably present.

### 2.5 **[NEW]** And used as a retrieval key, the graph matches the trained model

Neighbor average over training strains, retrieving on the gene's adjacency row (a
common-neighbors similarity), k swept on validation, all four split seeds, identical metric
and partition to the trained arms:

| retrieval key | val | sd | test | sd | val coverage |
|---|--:|--:|--:|--:|--:|
| oracle (own profile) | 0.6293 | 0.0103 | 0.6455 | 0.0112 | 1.00 |
| string12_0_experimental | **0.1964** | 0.0160 | 0.1745 | 0.0302 | 1.00 |
| union of all nine | 0.1889 | 0.0203 | **0.2013** | 0.0195 | 1.00 |
| string12_0_coexpression | 0.1877 | 0.0161 | 0.1809 | 0.0207 | 1.00 |
| string12_0_database | 0.1606 | 0.0188 | 0.1764 | 0.0113 | 0.74 |
| physical_interaction | 0.1485 | 0.0145 | 0.1895 | 0.0201 | 1.00 |
| cooccurence + database | 0.1448 | 0.0123 | 0.1648 | 0.0183 | 0.81 |
| **prot_T5** | 0.1052 | 0.0183 | 0.1269 | 0.0334 | 1.00 |
| tflink | 0.0408 | 0.0125 | 0.0104 | 0.0225 | 0.97 |
| regulatory_interaction | 0.0375 | 0.0276 | 0.0362 | 0.0156 | 0.98 |
| string12_0_neighborhood | 0.0313 | 0.0141 | 0.0112 | 0.0180 | 0.31 |
| string12_0_cooccurence | 0.0295 | 0.0100 | 0.0394 | 0.0125 | 0.33 |
| every rewired control | -0.005 to +0.023 | | -0.010 to +0.027 | | |

Paired within split seed against ProtT5, **positive on 4 of 4 seeds in every case**, on both
validation and test: `string12_0_experimental` +0.091 +/- 0.030 val, +0.048 +/- 0.024 test;
union +0.084 +/- 0.036 val, +0.074 +/- 0.014 test; `string12_0_coexpression` +0.083 +/- 0.033
val, +0.054 +/- 0.020 test; `string12_0_database` +0.055 +/- 0.028 val, +0.050 +/- 0.023
test; `physical` +0.043 +/- 0.032 val, +0.063 +/- 0.036 test.

Paired against each graph's own degree-preserving configuration-model rewiring:
`string12_0_experimental` **+0.182 +/- 0.016**, `string12_0_coexpression` +0.170 +/- 0.024,
`string12_0_database` +0.166 +/- 0.029, `physical` +0.126 +/- 0.019, and for the two
TF-target graphs only +0.024 +/- 0.012 and +0.015 +/- 0.010. Every rewired control sits
between -0.005 and +0.027, so this is topology and not degree.

Caveats stated plainly. k is chosen on validation, which is one degree of freedom, so test
is quoted alongside and agrees. The trained model's 0.1965 is a max over 9,900 epochs and
is itself an upward-biased order statistic, so the comparison if anything favors the model.
`string12_0_coexpression` is built from public mRNA compendia and could in principle share
provenance with the Kemmeren panel, but `string12_0_experimental` (primary interaction
evidence), `physical_interaction` (SGD curated PPI) and `string12_0_database` (curated
pathways) are expression-independent and carry the result equally, so the conclusion does
not rest on the one channel with a provenance question. Coverage is exact: the union and
the four dense graphs reach every validation strain.

The two graphs with the highest **per-edge** lift (co-occurrence 4.79x, database 3.59x) are
not the best **retrieval keys**, because co-occurrence reaches only 32% of validation genes.
Lift and coverage trade off, and the union buys both.

### 2.6 So: is it a set transformer with a perturbation token?

**Yes, and the model file says so itself.** From the `PerturbationGraphPropagation`
docstring, `torchcell/models/equivariant_cell_graph_transformer.py:740-770`, with the
numerical verification in the same docstring:

> the encoder runs at batch 1 on the WILDTYPE graph, so `H_genes` and `h_CLS` are identical
> for every strain; the only strain-dependent step is `EquivariantPerturbationTransform`,
> whose K/V set is just the |S_b| perturbed tokens. For a SINGLE deletion the softmax is
> over one key, so the attention weight is exactly 1 for every query gene and the attended
> vector is query-INDEPENDENT ... Verified numerically (max |attended_i - attended_0| = 0).
> ... The nine graphs shape h_i but never CARRY the deletion.

For ~96% of the training rows the model is exactly

  yhat(strain b, reporter i) = R(h_i, c_b),  c_b = W_O W_V h_{p(b)} in R^90,

with `h_i` identical for every strain. It is a permutation-equivariant set encoder over
frozen sequence embeddings with one strain-level 90-d vector added to every token; the
graph appears in one of six attention layers, symmetrized, with the KL off, on a module that
never carries the deletion. **Calling the incumbent a set transformer over gene embeddings
with a perturbation token is accurate, not a simplification.**

---

## 3. TF-target, complexes and pathways: what is there and what is absent

| mechanistically relevant relation | present? | where, and what it measures |
|---|---|---|
| TF -> target (curated) | **yes**, `regulatory_interaction` | SGD `regulation_details`, DiGraph, 37,703 directed edges, 52 reciprocal. Informative for propagation (0.5239), at chance for retrieval (0.85x) |
| TF -> target (compendium) | **yes**, `tflink` | TFLink v1.0, DiGraph, 200,643 directed edges, max degree 4,908. Best propagation signal in the graph set (0.5508); at chance for retrieval (0.98x) |
| YEASTRACT specifically | **no** | not ingested anywhere in torchcell |
| protein complexes | **not by name; the proxy is present and is among the best channels** | no CYC2008, no EBI Complex Portal, no `protein_complex` builder; `locus.py:324` has a passive `complexes: list[str]` nothing reads. `string12_0_database` folds curated complex and pathway co-membership into one score, and measures 3.59x / 13.01x lift and 0.169 val / 0.174 test as a retrieval key |
| GO co-annotation as a gene-gene edge | **no** | `G_go` is the ontology DAG, deliberately excluded from `SCEREVISIAE_GENE_GRAPH_MAP`; GO can only enter as an incidence graph, and v13 passes `incidence_graphs=None` |
| metabolic / pathway membership in fig3_core | **no** | `YeastGEM().bipartite_graph` attaches only when `per_metabolite` is an active head; v13's `active_heads` is `['per_gene']`. Verified: the built `cell_graph` carries exactly nine gene-gene edge types, no metabolite or reaction node type |
| co-expression from the training compendium (GEARS-style) | **no** | not built; `string12_0_coexpression` is STRING's cross-compendium channel, not this panel's |

### 3.1 Adding a TF-target edge type

Nothing needs adding. Both directed graphs are already in `cell_graph` with direction
intact; only `_build_head_mask` destroys it. The change is a config flag and about ten lines:
stop writing `head_mask[j,i]` for relations declared directed, and give each directed
relation a forward head and a reverse head. YEASTRACT would add a third curated TF-target
set, but TFLink is already a harmonized union of such resources and section 2.4 shows the
TF-target relation is useful for propagation and not for retrieval, so a new TF-target
source is not where the headroom is.

### 3.2 Adding a co-expression graph from the training compendium

Available today, leakage-free, and the campaign has explicitly ruled out the wrong version
of it. `expression_baselines.py`'s docstring says:

> each gene is deleted exactly once ... A perturbation representation derived from the
> deleted gene's own observed response (the obvious in-data choice) is therefore undefined
> for every val strain, and was measured to cover 0 of 155.

True for the gene's **row** (its own deletion profile). **Not** true for its **column**:
gene X appears as a *reporter* in every training strain, so `R_train[:, X]` is a
1,186-dimensional representation of X that touches no validation row and is defined for
90.3% of validation strains. The relation it encodes is the mechanistically right one: if
deleting Y moves X, X and Y are likely in the same process, so deleting X probably moves
what deleting Y moved.

**[NEW] I built it and measured it on all four v13 split seeds** (neighbor average, k swept
on validation):

| representation | dim | val | sd | test | sd |
|---|--:|--:|--:|--:|--:|
| oracle (own profile) | 6,169 | 0.6289 | 0.011 | 0.6455 | 0.011 |
| prot_T5 | 1,024 | 0.1073 | 0.019 | 0.1157 | 0.057 |
| prot_T5 + calm | 1,792 | 0.1073 | 0.024 | 0.1014 | 0.058 |
| prot_T5 + coexpr_svd32 | 1,056 | 0.1011 | 0.030 | 0.0971 | 0.019 |
| calm | 768 | 0.0844 | 0.014 | 0.0947 | 0.025 |
| **coexpr_col** (train-split co-expression) | 1,186 | **0.0710** | 0.008 | **0.0754** | 0.022 |
| coexpr_svd8 | 8 | 0.0644 | 0.004 | 0.0491 | 0.007 |
| random, matched width (seed 0) | 1,186 | -0.0057 | | +0.0072 | |

Paired across the four splits the fusion contrasts are null to slightly negative:
`prot_T5+coexpr_svd32 - prot_T5` = -0.006 +/- 0.017 val, -0.019 +/- 0.047 test;
`calm+coexpr_svd8 - calm` = +0.004 +/- 0.006 val, -0.011 +/- 0.030 test.

**The honest reading: the train-split co-expression representation carries real signal
against a matched-width random floor, and is NOT complementary to a protein language model
under neighbor averaging.** The naive GEARS-style co-expression input does not close the
gap. That is a measured null, and it should stop the obvious arm from being run at full
cost. Note it is also **well below** the interaction-graph adjacency row (0.071 vs 0.169 to
0.204): the panel's own 1,186 columns are a weaker gene representation than curated
interaction topology.

It is not null for everything. **[NEW]** In a ridge from the gene representation to the
top-32 coefficients of the train gene basis (split seed 0), adding the co-expression column
to ProtT5 raises the rank-8 reconstruction from 0.140 val / 0.079 test to **0.153 val /
0.092 test**, and per-component validation r on PC2 from +0.37 to +0.47, PC7 from +0.18 to
+0.25, PC8 from +0.26 to +0.28. One split, ridge chosen on validation, so suggestive only.

---

## 4. The oracle test **[NEW]**

The neighbor rule fixes the function class; only the retrieval key varies. Swapping the key
for the strain's own measured profile gives the ceiling of that function class.

| retrieval key | val | test |
|---|--:|--:|
| own measured profile (oracle) | **0.629 +/- 0.011** | **0.646 +/- 0.011** |
| union of the nine graphs | 0.189 +/- 0.020 | 0.201 +/- 0.020 |
| the trained CGT (n = 8, 9,900 epochs) | 0.1965 +/- 0.0222 | -- |
| ProtT5 | 0.107 +/- 0.019 | 0.116 +/- 0.057 |
| replicate ceiling of the label | 0.775 | |

**The best possible retrieval key is worth about 0.52 Pearson more than the best node
feature, and about 0.43 more than the graph union or the trained model.** The oracle uses
the held-out strain's own label, so it is a bound and not an achievable score.

Three follow-ups that locate the gap.

**(a) The oracle's advantage is not concentrated in the loud deletions.** Split seed 0
validation strains stratified by the norm of their response profile, k = 10:

| retrieval key | Q1 (quietest) | Q2 | Q3 | Q4 (loudest) | all |
|---|--:|--:|--:|--:|--:|
| oracle | +0.551 | +0.579 | +0.609 | +0.646 | +0.625 |
| prot_T5 | +0.010 | +0.076 | +0.088 | +0.126 | +0.090 |
| calm | +0.036 | +0.060 | +0.098 | +0.073 | +0.070 |
| coexpr_col | +0.016 | +0.042 | +0.016 | +0.079 | +0.045 |
| normalized_chrom_pathways | +0.009 | -0.001 | +0.033 | +0.029 | +0.021 |
| mean profile norm | 8.59 | 10.23 | 13.15 | 23.62 | |

The oracle works about as well on the quietest quartile as on the loudest; sequence
embeddings only work on the loudest and are at zero on the quietest. The failure is not
"most deletions do nothing".

**(b) The geometry, measured directly.** Over the 703,230 pairs of deleted genes in the
seed-0 training rows, the Spearman between representation cosine and deletion-profile
correlation is +0.018 (coexpr_col), +0.017 (prot_T5), +0.014 (calm), +0.012
(normalized_chrom_pathways), against a profile-correlation distribution with mean +0.009,
99th percentile +0.457, maximum +0.938, and 5.48% of pairs above +0.3. **The co-response
structure is real and no gene node feature is aligned with it.** Section 2.5 is the
resolution: the alignment lives in the edge set, not in any per-gene vector we hold.

**(c) Only two or three directions of the response are predictable at all.** Ridge from the
deleted gene's representation to coefficients on the train gene basis, split seed 0,
per-component validation r:

| representation | PC1 | PC2 | PC3 | PC4 | PC7 | PC8 | rank-8 recon val | test |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| prot_T5 | +0.33 | +0.37 | +0.24 | +0.03 | +0.18 | +0.26 | 0.140 | 0.079 |
| calm | +0.16 | +0.42 | +0.19 | +0.01 | +0.08 | +0.18 | 0.104 | 0.077 |
| prot_T5 + coexpr_col | +0.29 | +0.47 | +0.21 | +0.07 | +0.25 | +0.28 | 0.153 | 0.092 |
| coexpr_col | +0.15 | +0.27 | +0.03 | -0.01 | +0.22 | +0.10 | 0.082 | 0.073 |

Rank-r reconstruction ceilings with coefficients known
(`results/lowrank_output_ceiling.json`): rank 1 = 0.266, rank 2 = 0.412, rank 4 = 0.525,
rank 8 = 0.579, rank 16 = 0.666, rank 32 = 0.727. A *perfect* rank-2 placement already beats
the trained model (0.412 vs 0.197), and the genotype currently resolves PC1 and PC2 at
r ~ 0.3 to 0.45 and essentially nothing past PC3.

---

## 5. Would an expression-derived gene embedding close it?

**Measured, on the version available today: no.** Section 3.2's `coexpr_col` is exactly the
leakage-free SVD-of-the-train-compendium embedding, reaches 0.071 val / 0.075 test, and
fuses null against ProtT5. The repo's other evidence points the same way:

- `normalized_chrom_pathways` is the **best** representation on morphology in the kNN probe
  (0.100, above every sequence embedding) and near the floor on expression (B3 val 0.033,
  test 0.002). Functional annotation transfers to morphology and not to the transcriptome.
- `string12_0_coexpression` is a cross-compendium co-expression graph. Its *node feature*
  form is not built, but as an edge set it is one of the strong retrieval keys of section
  2.5 (0.194 val), which says the useful content is pairwise and not per-gene.
- Ahlmann-Eltze, Huber and Anders 2025 (doi:10.1038/s41592-025-02772-6) found no deep model
  beating simple baselines, and their linear model beat GEARS **while using GEARS's own
  perturbation embeddings**. GEARS's inputs are a GO similarity graph and a co-expression
  graph; the benchmark's finding is that those inputs carry no more than a bilinear map can
  use. Section 2.5 is consistent and sharper: the graph content is worth about what the
  trained model already gets, and a parameter-free rule extracts it.

Untested and honestly open:

- **A pretrained compendium foundation embedding** (Geneformer, scGPT, or a yeast model
  trained on SPELL). None is in the builder. SPELL is documented in
  `notes/torchcell.datasets.scerevisiae.spell.md` with a working PCL loader, and **is not
  built on this machine** (no `data/spell` under DATA_ROOT). Hypothesis (untested): an
  embedding over ~200 yeast studies encodes co-regulation that neither a protein LM nor
  this panel's 1,186 columns can.
- **A functional-profile embedding from another assay.** The candidates are already built
  LMDBs: the SGA genetic-interaction profile (`dmi_costanzo2016`) and the chemical-genomic
  fitness profiles (`env_chemgen_hillenmeyer2008`, `env_chemgen_hoepfner2014`, and six
  others). Hypothesis (untested): the SGA profile is the canonical functional gene
  embedding in yeast and should align with deletion-response similarity far better than
  sequence does, for the same reason SGA profiles cluster genes into pathways. Section
  4(b)'s rho is the exact cheap gate.

---

## 6. Proposed input-side arms, with cost and hypothesis

Ordered by (measured evidence behind it) / cost.

### Tier 0: CPU probes, hours, no GPU

**A1. Promote the graph-retrieval baseline to a committed script and add it to the baseline
table (B4).** Section 2.5 is currently a scratchpad measurement, and it is the single most
consequential number in this report: a parameter-free rule on an input the model already
has matches the trained model. It belongs in `experiments/019-simb-multimodal/scripts/`
beside `expression_baselines_split.py`, run on all four split seeds plus the proteome
panel, with the rewired controls. **Cost: half a day of CPU.** If it survives, the
expression document's baseline section changes materially, and so does the claim the paper
can make about the transformer.

**A2. Retrieval-alignment gate for every candidate representation.** Extend section 4(b) to
the SGA interaction profile (`dmi_costanzo2016`), each chemical-genomic fitness profile, GO
semantic similarity, and any new embedding, reporting rho against the +0.017 ProtT5
baseline. **Cost: under a day of CPU.** Hypothesis (untested): the SGA profile reaches
rho > 0.05; anything below 0.02 should not get a GPU arm at all. **Run this before any GPU
arm below.**

**A3. Recompute the oracle and the graph-retrieval baseline on the proteome panel.** Same
scripts, `fig3_proteome`, `protein_abundance`. **Cost: an hour.** Hypothesis (untested):
the proteome oracle sits far below 0.63 because the label ceiling is 0.42 to 0.61, which
would say the queued proteome work has little headroom and should be reprioritized.

### Tier 1: cheap GPU arms with evidence already in hand

**B1. Graph adjacency as a node feature.** Add a `GraphAdjacencyEmbeddingDataset` to
`NodeEmbeddingBuilder` serving, per gene, either the 6,607-d binary union-adjacency row or a
128-d SVD of it, and put it in `node_embeddings` beside the sequence blocks. This gives each
token its own neighborhood *identity*, which the layer-1 mask does not: the mask constrains
where a token may look, it does not tell a token who its partners are. **Cost: one builder
class plus a config entry; 3 seeds x 2 arms at 1,400 epochs, ~6 Delta A40 runs, ~300
GPU-hours.** Evidence: section 2.5, where the parameter-free version of exactly this feature
reaches 0.198 val / 0.199 test against ProtT5's 0.101 / 0.125 and the trained model's 0.197.
Hypothesis (untested): this is the highest expected-value GPU arm on the table, because the
signal is measured, large, paired against a degree control, and currently unreachable by
the architecture.

**B2. Graph-kNN as an explicit prior, not only a feature.** Ensemble or residualize: fit the
model on the residual after the graph-neighbor average, or blend the two predictions with
one weight fit on validation. **Cost: near zero on top of B1, no new training if the blend
is done on existing checkpoint dumps in `val-predictions/` and `test-predictions/`.**
Hypothesis (untested): the model and the graph-kNN make partly independent errors, in which
case the blend exceeds both; if they are highly correlated that is itself the finding, that
the transformer has already learned the graph relation through the mask.

**B3. Directed attention mask.** Stop symmetrizing `regulatory_interaction` and `tflink`;
give each a forward and a reverse head. **Cost: a config flag plus ~10 lines in
`_build_head_mask`; 3 seeds x 2 arms.** Evidence: TF->target 0.5508 vs symmetric 0.5057 vs
control 0.5017. Hypothesis (untested): the gain is small, since a +0.05 AUC prior applied at
one of six layers is weak, but it is the only graph-channel change with a positive data-side
measurement behind it and it is nearly free.

**B4. Mask-off (`P_free`) control, finally run.** Nine heads at layer 1 unconstrained.
**Cost: 3 seeds, 1,400 epochs, 3 runs.** Evidence: the mask's target is at chance on all
nine graphs; the mask has never been ablated in the expression campaign; it was adopted on
a throughput argument with an n = 1 accuracy comparison inside the noise. **Worth running
even if null**, because every graph claim in the expression document currently rests on an
unablated component. Note the tension this arm resolves: section 2.5 says the graphs carry
a great deal, and section 2.3 says the mask's target carries nothing; if `P_free` is null,
the reading is that the mask is an inert way of consuming a very informative input.

**B5. Retire `E_full` to `E_calm_ptt5`.** Drop `fudt_upstream` and `fudt_downstream`.
**Cost: zero, a config edit, and it makes every later run cheaper** (3,328 -> 1,792 input,
5.85M -> 1.8M preprocessor parameters). Evidence: `E_full - E_calm_ptt5` = -0.002 +/- 0.006
paired over 3 seeds; `model_stack` B3 test 0.091 vs `prot_T5` 0.119 over 4 splits. Not an
arm, a correction.

### Tier 2: input arms that target the measured gap

**C1. SGA genetic-interaction profile as the perturbation representation.** Concatenate the
deleted gene's Costanzo/Kuzmin interaction-profile vector (already an LMDB) to the token.
**Cost: a builder entry plus a coverage audit, since the SGA query set does not cover all
6,607 genes and the missing-value policy must be declared rather than imputed silently;
then 3 seeds x 2 arms.** Gated on A2. Hypothesis (untested): both the SGA profile and the
deletion transcriptome answer "what pathway is this gene in", read out through a deletion,
so this should align with deletion-response similarity better than sequence.

**C2. Reporter-side graph and co-expression token.** Sections 2.5 and 3.2 both used their
representations on the *perturbation* side. The same vectors exist for every reporter and
the reporter side has never been tested (`graph-prior-probe.md:157-159` lists it as open
hypothesis 2). Give each gene token `[sequence ; adjacency row ; train-split response
column]`. **Cost: builder entries; the response column is split-dependent and must be
rebuilt per split seed, which is a real provenance obligation; 3 seeds x 2 arms.**
Hypothesis (untested): helps more than the perturbation-side version did, because the
per-gene head currently learns each reporter's responsiveness from scratch while the
information sits in the training matrix.

**C3. Yeast expression foundation embedding from SPELL.** Build the compendium (the loader
exists), fit an embedding over ~200 studies with Kemmeren and Sameith excluded, use it as a
gene representation. **Cost: a dataset build plus embedding training, on the order of a week
of engineering before any GPU arm.** Hypothesis (untested): an external compendium carries
co-regulation this panel's 1,186 columns cannot, the one form of expression-derived
embedding section 5's null does not cover.

### Tier 3: structural

**D1. Put the deletion inside the encoder.** The measured additivity
(`equivariant_cell_graph_transformer.py:740-770`, max |attended_i - attended_0| = 0) means
no readout or objective change can enlarge the hypothesis class while the perturbation is a
single post-encoder key. Already queued in `3-next.tex`; section 2.4 adds a second argument,
that the TF->target relation is the one graph relation that *does* predict which reporters
move, and it can only be used by a mechanism that routes the deletion along edges, which
`perturbation_propagation.enabled: false` currently forbids.

**D2. Learn the retrieval metric instead of the profile.** Train a head on
`(gene_i, gene_j) -> profile correlation` over training pairs, on the representations A2
admits plus the graph, and use the learned similarity for a neighbor average. **Cost: CPU or
one GPU-day; a far smaller learning problem than the full profile.** Hypothesis (untested):
703k labeled pairs against 1,186 training profiles is much better posed than the current
objective, and section 4 puts its ceiling at 0.63 rather than 0.20.

---

## 7. What I would not do

- Do not build the `P_graph` arm (a pair term as a function of network distance): the
  network does not carry that relationship in the propagation sense
  (`graph_prior_probe.md`). Section 2.5 is not a counterargument; it is about retrieval.
- Do not spend a round on more nucleotide or promoter representations. Six NT windows and
  two species-LM flanks are at or below the random floor on both baselines and on the
  neighbor probe, and v11's `E_full` measured them at -0.002 inside the model.
- Do not run a full-cost GPU round on the naive GEARS-style co-expression input. Section
  3.2 measured it, and the interaction graph beats it by a factor of two and a half as a
  retrieval key.

---

## 8. One thing the review should settle

Section 2.5 implies the expression document cannot currently claim the transformer beats
its baselines on this task, because the strongest baseline has not been run. A neighbor
average on the graph union reaches 0.189 +/- 0.020 validation and 0.201 +/- 0.020 test
against the incumbent's 0.1965 +/- 0.0222 (a max over 9,900 epochs). Either A1 overturns it,
or the paper's framing changes from "the model learns the perturbation response" to "the
interaction graph determines it, and the model has not yet exceeded what the graph gives for
free". That is a claim about the paper, not about what to run next, and it should be decided
before the next round is launched.

---

## Scripts I ran for the [NEW] numbers

Read-only, in the session scratchpad, not written into the repo. Any number used in a note
or the paper must first come from a committed script in the experiment folder.

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/oracle_embed.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/oracle_embed2.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/pc_predictability.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/retrieval_structure.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/edge_profile_similarity.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/graph_as_embedding.py
