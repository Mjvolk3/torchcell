# Agent 03 -- Architecture audit: "are genes allowed to interact enough?"

Scope: the model as ACTUALLY CONFIGURED for the v13 expression rounds and the v14 proteome
round. Everything below traces to a file+line, a resolved config key, a W&B run, a committed
results JSON, or a CPU diagnostic I ran in this session (scripts left under
`/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/`).

---

## 0. The short answer to the question asked

**No, and not in the way the campaign has been assuming.**

Genes interact *plenty* in the encoder: five of six encoder layers run unmasked dense
attention over all 6,608 tokens, so every gene sees every gene in one hop. But the encoder
runs at **batch 1 on the wildtype graph, BEFORE the perturbation exists**, so none of that
interaction is strain-dependent. After the deletion is injected, the only gene-gene channel
is a 32-latent Perceiver bottleneck. And the deletion itself enters as **one 90-dimensional
vector added identically to all 6,607 gene tokens**.

The sharper finding is that "how much do genes interact" is not the binding constraint at
the current score. Two measurements I ran settle it:

1. The model's val prediction matrix is **effectively rank 4.8** and a rank-5 truncation
   reproduces its entire score, while a *rank-1 oracle* on the same data scores 0.317
   against the model's 0.223. The model is worse than a rank-1 oracle. Output rank is not
   what is costing it points right now.
2. The model completely fails the single most predictable number in every profile: the
   deleted gene's own log2 ratio, which sits **8.3 sd below its own column mean** and is the
   largest |change| in the true profile for the median strain. The model puts it at the
   **49.5th percentile** of its own predicted |change| (chance) and predicts **+0.036** where
   the truth averages **-2.42**.

---

## 1. The configuration that actually runs (resolved, not inherited by eye)

`cgt_expr_v13_split` composes `v12_head -> v11_emb -> v9_mask -> 012 -> 011 -> 010 -> 008 ->
006 -> embed_005 -> decoder_004 -> decoder_003 -> default`. Resolved with hydra `compose`:

| key | value |
|---|---|
| `model.hidden_channels` | 90 |
| `model.num_transformer_layers` | 6 |
| `model.num_attention_heads` | 9 |
| `model.dropout` | 0.1 |
| `cell_dataset.node_embeddings` | `[fudt_upstream, calm, prot_T5_all, fudt_downstream]` = 3,328-d |
| `model.learnable_embedding.enabled` | **false** (content features only) |
| `model.attention_mask.enabled` / `.layers` | **true** / **`[1]`**, heads 0..8 -> the nine graphs |
| `model.graph_regularization.graph_reg_lambda` | **0.0** (KL off; the mask replaced it) |
| `model.perturbation_head.{num_layers, ffn_mult, residual, num_heads, pooling}` | 1, 4, **postln**, 9, sum |
| `model.perturbation_head.{null_sink, hadamard}` | **false**, **off** |
| `model.perturbation_propagation.enabled` | **false** |
| `model.cross_gene.enabled` | **false** |
| `model.post_perturbation_mixing` | **enabled**, `num_latents: 32`, `gate_mode: on` |
| `model.observed_labels` | **enabled**, `gate_mode: on` |
| `multitask.mask_schedule` | **`[0, 10, 100, 1000]`** |
| `multitask.{per_gene_weight, concat_context, response_basis_rank, bilinear_rank, film_on_pert_set, context_readout, linear_readout, free_gene_dim}` | all **off / 0** for the `V_ref` arm (`concat_context: true` is the only change in `V_concat`) |
| `multitask.dist` | quantile, K=19, pinball |
| `data_module.batch_size` / `lr` / `weight_decay` / scheduler | 32 / 3e-4 / **1e-8** / **none, no warmup** |
| `trainer.max_epochs` | 6,000 |

So: **every pair-term mechanism ever built into this file is OFF in the round that produced
the headline numbers.** The propagation module, the null sink, the Hadamard operator, the
response basis, the bilinear features, FiLM, the GEARS per-gene row, the State context row,
the GEARS pooled cross-gene state: all disabled. The only two non-baseline blocks running are
the observed-label encoder and the 32-latent Perceiver, both added for the v9 masked objective.

---

## 2. The computational path: "gene p is deleted" -> "predicted log2 ratio of gene j"

File: `torchcell/models/equivariant_cell_graph_transformer.py`.

**Step 1 -- gene tokens (strain-invariant).** `forward` `:2895-2924`. `cell_graph["gene"].x`
is a FIXED `[6607, 3328]` matrix (four concatenated embeddings). It goes through
`embedding_preprocessor`, a 2-layer MLP `Linear(3328, 1709) -> LayerNorm -> GELU -> Dropout ->
Linear(1709, 90) -> LayerNorm -> Dropout` built at `:2178-2213`. Output `gene_embs [6607, 90]`.

**Step 2 -- encoder (strain-invariant).** `:2926-2991`. `X = [cls; gene_embs].unsqueeze(0)`,
shape `[1, 6608, 90]`, i.e. **batch 1**. Six `GraphRegularizedTransformerLayer` blocks
(`:34-183`). Because `graph_reg_lambda == 0` and `return_attention` is False, every layer
takes the fused SDPA path (`:117-140`). Layer 1 receives `head_mask` (all nine heads masked to
their graph's symmetrized adjacency + self-loop + CLS, `_build_head_mask` `:2582-2623`).
**Layers 0, 2, 3, 4, 5 are unmasked**: full dense attention over all 6,608 tokens.
`H_genes [6607, 90]` and `h_CLS [90]` are therefore **identical for every strain in the
dataset**, and identical for every strain ever seen.

**Step 3 -- the perturbation, and this is the whole strain-conditional path.**
`EquivariantPerturbationTransform.forward` `:647-738`, called at `:2999`. One
`nn.MultiheadAttention(90, 9)` with `query = H_cur [1, 6607, 90]` (every gene) and
`key = value = H_cur[pert_idx_b]`, i.e. **only the |S| perturbed tokens**. `num_layers = 1`.
Then post-LN residual `norm1(x + dropout(attended))` -> FFN(4x) -> `norm2` (`_apply_residual`
`:635-645`).

At `|S| = 1` (95.4% of the expression build) the softmax is over one key, so `alpha = 1.0`
identically and the attended vector is **query-independent**: `c_b = W_O(W_V h_p + b_V) + b_O`,
one 90-d vector added to all 6,607 tokens. Measured in
`results/perturbation_selector_degeneracy.json`: across-query spread `3.31e-09`, and re-drawing
`W_Q`/`W_K` at std=10 changes the output by **exactly 0.0**.

So `H_pert[b, i] = g(h_i + c_b)`: gene identity and strain identity meet **exactly once, by
addition**, and the next operation is a LayerNorm.

**Step 4 -- observed labels (k > 0 only in training).** `:3027-3031`,
`ObservedLabelEncoder` `:941-1018`. `proj([value*mask, mask]) : R^2 -> R^90`, added per gene.
*Precision note:* the docstring claims a fully masked forward is "an identity". It is not.
`proj` has biases, so at k=0 it adds a learned **constant** `W2 ReLU(b1) + b2` to every gene
token of every strain. Uniform across genes and strains, so it cannot affect
`pearson_per_feature`; but "identity" is the wrong word and the checkpoint confirms nonzero
biases (`observed_label_encoder.proj.0.bias` mean |.| = 0.217).

**Step 5 -- the ONLY post-perturbation gene-gene channel.** `:3040-3041`, `PerceiverMixing`
`:1229-1252`. `write_attn(query=32 latents, key=value=H_pert[b])` pools all 6,607 genes into
32 latents; `read_attn(query=H_pert[b], key=value=latents)` reads them back. Gate is a
non-trainable buffer fixed at 1.0 (`gate_mode: on`).

This block is the one place the current architecture DOES create a genuine pair-(p, i) term:
`q_i` depends on `h_i` and `c_b`, the latents depend on `c_b`, and the softmax over latents is
not an additive function of the two. **The rank ladder in
[[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]] ("additive =
rank 0") is stale for v9 onward**: it was written before `post_perturbation_mixing` became a
forced-on default.

**Step 6 -- the readout.** `PerGeneHead` `:1768-1833`, called at `:3112`. For `V_ref` this is
`Linear(90, 90) -> ReLU -> Dropout -> Linear(90, 19)`, a **shared MLP applied to each gene
token independently**, 9,919 parameters. `point()` takes the median knot
(`torchcell/losses/distributional.py:919-931`). `_gather_predictions` then selects the 6,127
measured-gene columns out of 6,607.

**Reachability.** Every one of the 6,127 output genes is reached from the deleted gene in a
single step, through `c_b`. The graphs never carry the deletion: the encoder ran before it.
The "influence(X -> Y) ~ (A^L)_{YX}" derivation in
[[experiments.019-simb-multimodal.graph-prior-probe]] section 3 describes an architecture
that **is not implemented**. There is no hop structure between the deletion and the readout.

Corollary: the layer-1 mask restricts one of six layers of a strain-invariant encoder, toward
a target that `results/graph_prior_probe.json` measured **at chance on all nine graphs**
(best excess over the degree control +0.0046). Three of the nine masked heads are additionally
near-inert: 29% / 46% / 46% of deleted genes have any edge at all in
`string12_0_neighborhood` / `fusion` / `cooccurence`, so those heads see only self + CLS for
most genes.

---

## 3. Capacity accounting (measured on the real checkpoint)

`best_metric.ckpt` for run `wq8y8nd5`, epoch 3783, global_step 147,576.

| component | trainable params | share |
|---|---:|---:|
| `embedding_preprocessor` (3328 -> 1709 -> 90) | **5,846,759** | **87.6%** |
| `transformer_layers` (6 x 98,370) | 590,220 | 8.8% |
| `post_perturbation_mixing` (Perceiver, M=32) | 101,430 | 1.5% |
| `perturbation_transform` (the ONLY strain-conditional block) | **98,370** | **1.5%** |
| `perturbation_head` (built, unused for expression) | 16,381 | 0.25% |
| `per_gene_head` `V_ref` | **9,919** | **0.15%** |
| `observed_label_encoder` | 8,460 | 0.13% |
| cls_token | 90 | ~0 |
| **total** | **6,671,629** | |

(The raw `state_dict` reports 6,782,272 because `perturbation_transform` registers
back-compat aliases `.cross_attn/.ffn/.norm1/.norm2` onto layer 0 at `:601-604`, double
counting 98,370. `.parameters()` dedupes; 6.67M is the trainable count.)

**96.4% of the model is strain-invariant. 1.5% is the entire genotype-conditional path.
0.15% is the readout.** `V_concat` moves the head to 26,119 (0.39%).

Other capacity facts:
- d_model 90, 6 layers, 9 heads, head_dim 10.
- Effective receptive field: **1 hop, all 6,607 genes**, in the encoder; **1 step, all genes,
  via one 90-d vector**, for the perturbation.
- Strain-context map `C = W_O W_V` from the checkpoint: 90x90, participation-ratio effective
  rank **25.1**, top-5 singular directions carry 33% of its energy.
- `embedding_preprocessor` `W1` eff rank 36.2, `W2` eff rank 54.3.
- Attention projection norms in the perturbation block: `||W_Q|| 19.22`, `||W_K|| 16.93`,
  `||W_V|| 6.23`, `||W_O|| 4.89`, against an init of ~6.7 for all of Q/K/V. Q and K **grew
  2.5-2.9x** while V shrank. Hypothesis (untested): the 4.6% of rows with |S| = 2 are the only
  gradient source for Q/K, and with weight decay 1e-8 there is nothing to stop them saturating
  the 2-key softmax into a near one-hot selector, i.e. the model may be learning to pick ONE
  gene of a double. 15 doubles in the val dump is too few to test here.

---

## 4. Where scale information is destroyed

Three sites, in order of how much they matter.

1. **Post-LN immediately after the additive fusion** (`_apply_residual` `:635-645`).
   `LN(h_i + c_b)` is invariant to positive rescaling, and `||h+c||^2 = ||h||^2 + 2<h,c> +
   ||c||^2`, so the inner product `<h_i, c_b>` (the natural "how related is reporter i to the
   deleted gene p" term) is removed by the operation immediately following the only place the
   two ever meet. This is the note's claim and it is correct as written.
2. **Attention dropout at |S| = 1, which nobody has flagged.** `nn.MultiheadAttention(...,
   dropout=0.1)` applies dropout to the attention WEIGHTS. With one key the weight is exactly
   1.0, so during training **10% of gene tokens receive `attended = 0`** (no perturbation
   context at all) and the other 90% receive `c_b / 0.9`. At eval, none. The only
   strain-conditional signal in the model is being randomly deleted per gene per step and
   rescaled by 1.111. Config-only to test: `model.perturbation_head.dropout=0.0`.
3. **The pinball/MAE objective de-weights exactly the outliers that carry the pair
   information.** Measured on the val target matrix: the 170 deleted-gene cells are 0.0179%
   of cells but carry **3.41% of the squared deviation** and only **0.38% of the absolute
   deviation**. An MSE-family loss weights them **8.9x more** than the pinball median does.

---

## 5. Diagnostics I ran on the existing dumps and checkpoints (all CPU, minutes)

### 5.1 The prediction matrix is effectively rank 5, and worse than a rank-1 oracle

`/scratch/projects/torchcell-scratch/val-predictions/compute-0-2-2397312_3fcf....json`
(`V_ref_s0` seed 0, split 0, 155 val strains x 6,127 genes, best-metric checkpoint).
`pearson_per_feature` recomputed from the dump = **0.2227**.

| prediction truncated to rank | var frac | pearson_per_feature |
|---:|---:|---:|
| 1 | 0.416 | 0.1303 |
| 2 | 0.533 | 0.1646 |
| 3 | 0.645 | 0.2157 |
| **5** | 0.745 | **0.2280** |
| 10 | 0.860 | 0.2205 |
| 154 (full) | 1.000 | 0.2227 |

A **rank-5** truncation slightly BEATS the full-rank prediction. Participation-ratio effective
rank of the centered prediction matrix = **4.80**; of the target = 13.01 (bounded by 155
strains; the campaign's own residual-covariance estimate is 32.78).

Oracle comparison, same val matrix (best rank-k reconstruction of the TARGET, scored as a
predictor): rank-1 **0.317**, rank-2 0.414, rank-3 0.504, rank-5 0.573, rank-10 0.686,
rank-33 0.861. **The model at rank ~5 scores 0.223 against a rank-1 oracle's 0.317.**
So the binding constraint at today's score is the genotype -> amplitude map, not output rank.
(Output rank becomes binding later: `results/lowrank_output_ceiling.json` train-basis ceilings
are 0.525 at rank 4, 0.579 at rank 8, 0.726 at rank 32.)

Subspace alignment: the prediction's top gene-space direction aligns with the target's top
direction at cosine **0.83**, but only **34%** of target variance lies in the prediction's
top-5 gene subspace. The model has found the dominant knockout response program and a
per-strain amplitude on it, and little else.

Proteome (v14) is the same picture, harder: prediction effective rank **1.35 to 2.11** against
a target effective rank of **36.5 to 47.2**. The proteome model is a single global strain
factor times a per-protein loading.

Under-dispersion: per-gene prediction sd median **0.0452** vs target **0.1439** (3.2x).

### 5.2 The deleted gene's own value: the model is at chance on the largest signal in the data

Correct column alignment matters here. `head_keys` in the dump is the RAW 6,169-key list while
the matrix is the 6,127 kept columns (`build_head_alignments`
`experiments/019-simb-multimodal/scripts/train_cgt_multitask.py:592-605` drops keys absent
from `node_ids`). Aligning by `[k for k in keys if k in gene_set]` reproduces 6,127 exactly.

| quantity | value |
|---|---|
| deleted gene's own target, mean +- sd | **-2.417 +- 1.336** (n = 170) |
| same, as z within its own gene column | **-8.26 +- 3.70** |
| all-cell target, mean +- sd | +0.033 +- 0.220 |
| deleted gene's own PREDICTION, mean +- sd | **+0.036 +- 0.175** |
| corr(self prediction, self target) | **0.055** |
| percentile of the deleted gene in the strain's \|TRUE change\| (0 = largest) | **0.000** (median) |
| percentile of the deleted gene in the strain's \|PREDICTED change\| | **0.495** (median) |

Post-hoc substitution, nothing retrained:

| substitution at the 170 self cells | val pf | test pf |
|---|---:|---:|
| none (as trained) | 0.2227 | 0.1095 |
| **constant -1.0** | **0.2366** | **0.1241** |
| constant = the observed mean (-2.42 val / -2.44 test) | 0.2376 | 0.1256 |
| oracle (true value) | 0.2382 | 0.1260 |

**+0.0139 on val and +0.0146 on test from a single constant at one cell per strain.** That is
larger than every architecture arm this campaign has resolved except `H_concat` (+0.0136 at
matched budget). Proteome gets only +0.0017 because just 19 of 113 deleted proteins are
measured.

The hop-0 self-indicator that supplies exactly this feature **exists in the code**
(`PerturbationGraphPropagation`, `num_features = len(graphs)*hops + 1`, `:813`) and is
**disabled** in every v9-v16 config. The one time it was tried (`A2_self`, wave 1) it ran at
`gate_mode: rezero` with the gate closed at init and gate logging not yet implemented, at
n = 1 seed, scoring -0.0010 smoothed. That is not a measured null; the mechanism's engagement
was never verified.

Correction to a standing claim: the argument in
[[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]] section 4, that
zeroing the deleted gene's row "changes exactly ONE prediction of 6607, so ~99.98% of scored
predictions do not move", is arithmetically right and **operationally wrong**. That one
prediction is worth +0.014 on the leaderboard metric because the metric is a mean of per-gene
correlations and each gene column has exactly one cell with an 8-sigma outlier in it.

### 5.3 The masked objective is being paid for and then unlearned

Run `wq8y8nd5`, full history:

| epoch | val@k0 | val@k3 (1000 revealed) | traineval@k0 | val/loss |
|---:|---:|---:|---:|---:|
| 0 | 0.007 | 0.036 | 0.004 | 0.2645 |
| 101 | 0.049 | 0.268 | 0.044 | 0.2520 |
| 201 | 0.091 | 0.385 | 0.186 | 0.2447 |
| 401 | 0.074 | 0.400 | 0.338 | **0.2430 (min)** |
| 801 | 0.163 | 0.475 | 0.570 | 0.2455 |
| **1201** | 0.191 | **0.482 (peak)** | 0.674 | 0.2509 |
| 2001 | 0.201 | 0.418 | 0.728 | 0.2588 |
| 3001 | 0.198 | 0.351 | 0.758 | 0.2668 |
| 4071 | 0.199 | **0.317** | 0.773 | 0.2720 |

Readings:

- **val@k0 is FLAT from epoch 2000**, not "still rising at 4000". Max 0.2138 at epoch 3641;
  0.2011 at 2001, 0.1976 at 4001. Epochs 2000-4000 bought nothing on the headline metric.
- **The imputation capability peaks at epoch ~1200 and is then destroyed** (0.482 -> 0.317,
  a 34% loss). It is actively unlearned as the model memorizes.
- **On train the revealed labels are ignored entirely**: at the final epoch
  `traineval@k0 = 0.7729` and `traineval@k3 = 0.7740`. Revealing 1,000 true values changes
  the train fit by **+0.001**. The conditioning pathway receives almost no gradient because
  the unconditioned path already fits the training strains.
- **The channel is far below a linear oracle.** `results/masked_conditioning_oracle.json`
  (ridge from the revealed values to the rest, val): m=10 -> **0.408**, m=100 -> **0.676**,
  m=1000 -> **0.793**. The model: k1 (10) **0.215**, k2 (100) **0.250**, k3 (1000) **0.326**
  at its final checkpoint. *A ridge regression on 10 revealed genes beats the trained model
  on 1,000.* (Caveat: the oracle reveals a fixed gene set per draw while
  `_observed_feature_mask` `:1189-1203` draws a different set per strain, so the model's task
  is somewhat harder. The gap is far too large to be that.)
- Plausible mechanism, unverified: `PerceiverMixing.write_attn` softmaxes over all 6,607 gene
  keys, so 10 revealed genes are 0.15% of the key set and their contribution to the latents is
  diluted ~660x unless attention learns to find them.

Cost accounting for the masked objective, from the code: `_masked_step` `:1246-1247` runs an
extra `torch.no_grad()` forward every step to decode targets, and `:1260-1261` samples ONE k
uniformly from `[0, 10, 100, 1000]`, so **only 25% of gradient steps are the k=0 task the
leaderboard measures**, and each step costs 2 forwards instead of 1. Removing the schedule
would give roughly `4 x (2F+B)/(F+B) ~ 6x` more on-task gradient per wall-clock hour
(arithmetic estimate; the forward/backward cost ratio is assumed, not measured).

### 5.4 The proteome plateau is the same disease, earlier

Run `uc0pm2pv` (P_ref_s0 seed 0, 4,476 strains, 1,850 proteins):

| epoch | val@k0 | val@k3 | traineval@k0 |
|---:|---:|---:|---:|
| 51 | 0.064 | 0.350 | 0.055 |
| 101 | 0.104 | 0.459 | 0.153 |
| **181** | **0.1298 (max)** | ~0.50 | ~0.30 |
| 301 | 0.103 | 0.541 | 0.482 |
| 501 | 0.095 | 0.561 | 0.564 |

The genotype channel peaks at epoch 181 and decays while the label-conditioning channel keeps
improving to 0.56. That is as clean a statement as the data can make that the bottleneck is
the **genotype -> response map**, not the decoder and not the output factorization.
(Flag for the data agent: k3 = 0.56 exceeds the 0.42 duplicate-strain reliability ceiling
quoted in the shared context, which suggests the conditioning is partly reading same-plate
measurement structure rather than biology.)

### 5.5 One run in 24 never learned

`825on260` (V_ref_s0, seed 1, split 0): val@k0 and traineval@k0 both sit at ~0.00 from epoch 0
to 4001. A dead run. The same signature appears in v12: `kjs3u9rw` (H_basis64 s0) 0.0557,
plus 0.0576 and 0.0589 in `H_gears` and `H_pergene_basis64`, against 0/16 collapses in
`H_ref`/`H_concat`/`H_linear`/`H_state`. Hypothesis (untested): post-LN in the perturbation
block with no warmup, made more fragile by the added zero-gated branches. The config already
declares `regression_task.lr_scheduler.warmup_steps` and it is 0.

---

## 6. Correction to a standing claim about the v12 readout round

The shared context states "v12 was a readout round (per-gene head +0.041 on seed 0)". From
`results/head_round_readout.json`, arm means at the matched 1,399-epoch budget (n = 4 seeds):

| arm | mean | sd | vs H_ref |
|---|---:|---:|---:|
| H_concat | 0.1876 | 0.0053 | **+0.0136** |
| H_linear | 0.1765 | 0.0077 | +0.0025 |
| H_ref | 0.1740 | 0.0076 | -- |
| H_state | 0.1733 | 0.0070 | -0.0007 |
| **H_pergene** | 0.1656 | 0.0141 | **-0.0084** |
| H_basis64 | 0.1532 | 0.0661 | -0.0208 (2 collapsed runs) |
| H_gears | 0.1344 | 0.0523 | -0.0396 (2 collapsed runs) |

GEARS's per-gene row measured **negative** at the matched budget. The +0.04 figure is the
**500-epoch** paired contrast: `H_pergene` +0.0319 (4/4 positive) and `H_state` +0.0356
(4/4 positive), both of which wash out by 1,400. That is not a null result; it is a
**training-speed** result. Adding per-gene output parameters gets the model to the same place
three times faster, which is directly relevant to the "slow rise over thousands of epochs"
complaint.

---

## 7. Comparison to the published readouts

| model | how the perturbation reaches gene j | gene-gene interaction after conditioning | per-output-gene parameters |
|---|---|---|---|
| **GEARS** | perturbation embedding refined by a GNN over a GO similarity graph, summed, added to each gene embedding | a cross-gene MLP over all gene embeddings | **yes**: `w_u in R^d, b_u in R` per gene |
| **scGPT** (masked-value readout) | a condition token inside the same transformer | **full self-attention over gene tokens, after conditioning** | yes (per-gene decoder) |
| **CPA** | additive latent `z_basal + sum_p e_p` | none; the decoder MLP does the work | **yes**: decoder output layer is gene-indexed |
| **low-rank bilinear** (B2, measured 0.1040 on seed 0) | `a_i + sum_r u_r(h_i) v_r(c)` | none | yes (`a_i`) |
| **this model, as configured** | one 90-d vector added identically to all genes, from a cross-attention that is degenerate at \|S\|=1 | **32-latent Perceiver bottleneck only** | **NO** (shared MLP over tokens) |

The model has taken the union of the *weakest* readout choice from each comparator:

1. **CPA's additive latent without CPA's gene-indexed decoder.** CPA can afford an additive
   latent precisely because its decoder has one output row per gene. Ours composes to
   `F(h_i + c_b)` with a single shared `F`, so reporter identity enters only as a point in a
   90-d space, inside the same function the perturbation enters.
2. **GEARS's structure without GEARS's perturbation GNN.** `perturbation_propagation` is the
   analogue and it is disabled. Separately, `results/graph_prior_probe.json` measured all nine
   graphs at chance for predicting which reporters respond (largest excess over the degree
   control +0.0046), so the GEARS mechanism as specified would propagate over a graph that
   does not carry the relationship. The one place signal exists is **direction**: TFLink
   TF->target 0.5508 and regulatory TF->target 0.5239 against 0.5057/0.5055 symmetrized, and
   `_build_head_mask` `:2617-2618` symmetrizes, throwing it away.
3. **scGPT's masked objective without scGPT's post-conditioning attention.** The objective
   requires a channel that can carry an observed value from gene j to gene i. That channel is
   32 latents behind a softmax over 6,607 keys, and section 5.3 measures it delivering less
   than a 10-gene ridge.
4. **Below a rank-1 bilinear oracle.** The rank-1 oracle scores 0.317 and the model 0.223.
   The B2 bilinear ridge on gene embeddings scores 0.1040, so the model is 2x the smooth
   baseline, but it is not near what its own output structure could support.

---

## 8. What explains the three symptoms

**train 0.74 vs val 0.20.** The input to the strain-conditional path is a FIXED 3,328-d vector
per gene, and 87.6% of the model (5.85M parameters) is a 2-layer MLP mapping that fixed vector
to a 90-d token. Generalizing to a new deleted gene is exactly "interpolate a 5.85M-parameter
MLP at a new point in a 3,328-d space populated by 1,244 training points", and the only thing
regularizing that interpolation is weight decay **1e-8** and dropout 0.1. Memorization does
not even need the decoder: the preprocessor can act as a lookup table keyed on the gene row.
Meanwhile the objective's own arithmetic pushes the same way: 75% of steps train a conditioned
task the model solves by memorization, and the reward for learning the one genuinely
pair-dependent value (the self-gene) is de-weighted 8.9x by the pinball median.

**Slow rise over thousands of epochs.** Four contributions, each measured or arithmetic:
(a) only 25% of steps are the k=0 task; (b) each step pays an extra no-grad forward;
(c) the readout has no per-gene output parameters, and the v12 contrast shows adding them is
worth +0.032 to +0.036 at epoch 500 and nothing at 1,400, i.e. the shared MLP takes ~3x longer
to reach the same place; (d) no warmup and no schedule with post-LN in the one
strain-conditional block, which also produced 1 dead run in 24.

**Proteome early plateau.** Same architecture, 4,476 strains and 1,850 proteins. val@k0 peaks
at epoch 181 at 0.130 while traineval climbs to 0.60, and the prediction matrix collapses to
effective rank 1.35-2.11. With more strains the memorization route is reached faster, and the
genotype channel, which is 1.5% of the parameters and rank-degenerate at |S|=1, is abandoned
first.

---

## 9. Five arms, each with its hypothesis and a cheap diagnostic

Ordered by measured evidence per unit of cost. Arms 1, 2 and 5 are config-only.

### A1 -- Self-indicator: give the model "is gene i the one deleted"

**Change (config only):**
```
model.perturbation_propagation.enabled=true \
model.perturbation_propagation.hops=0 \
model.perturbation_propagation.gate_mode=on
```
`hops=0` makes `num_features = 1` (`equivariant_cell_graph_transformer.py:813`), the hop-0
indicator alone. Cost 8,460 parameters, no sparse matmuls, no adjacency dependence, transfers
to organisms with no interactome. `gate_mode=on` is mandatory: the only prior run of this arm
had a ReZero gate closed at init and no gate logging.

**Hypothesis:** the deleted gene's own column is 8.26 sd below its column mean and is the
largest |change| in the true profile for the median strain; the model is at the 49.5th
percentile on it (chance), r = 0.055.

**Diagnostic, already run:** post-hoc substitution of a constant -1.0 at those cells moves
val pf 0.2227 -> 0.2366 and test pf 0.1095 -> 0.1241. That is a measured lower bound on what
a working indicator buys, obtained from the existing dump with no training.

**Pair it with:** `multitask.dist=point` + MSE on a second arm, or a per-cell loss weight on
the self cell, because the pinball median de-weights this outlier 8.9x relative to MSE.

### A2 -- Stop paying for the masked objective on the expression head

**Change (config only):** an arm with `multitask.mask_schedule=null` (or `[0]`), and a second
arm with a non-uniform k sampler weighted 0.7 toward k=0 (a ~5-line change at
`train_cgt_multitask.py:1260-1261`).

**Hypothesis:** the k>0 branch adds +0.001 to the train fit, peaks on val at epoch ~1200 and
then loses 34% of its own capability, and consumes 75% of gradient steps plus an extra forward
per step. Removing it gives ~6x more on-task gradient per wall-clock hour, which directly
addresses "slow runs". The v9 design note already predicted the k0 score would not improve
from masking; nobody has measured whether it is COSTING the k0 score.

**Diagnostic, already run:** the `traineval@k0` vs `@k3` identity (0.7729 vs 0.7740) and the
val@k3 decay curve in section 5.3. A cheaper pre-check: rerun the same comparison on any two
other v13 runs to confirm the decay is not seed-specific.

### A3 -- Collapse the 5.85M-parameter gene preprocessor to a linear map

**Change (config only):** `model.learnable_embedding.preprocessor.num_layers=1`. The builder
at `:2191-2211` then emits `Linear(3328, 90) -> LayerNorm -> Dropout` = **299,610 parameters**,
a 95% cut; total model 6.67M -> 1.12M. Ladder it with `weight_decay` in {1e-8, 1e-4, 1e-2}.

**Hypothesis:** the generalization gap is dominated by an unconstrained, non-smooth map from
a fixed 3,328-d input to the 90-d token, fitted at 1,244 points. Forcing it linear makes
"similar embedding -> similar response" a structural property instead of something the
optimizer may or may not find. v10's 2^4 grid varied trunk depth/width and weight decay but
never the preprocessor depth, and `cgt_embed_005`'s header records that weight decay alone
moved val by 0.002.

**Diagnostic, runnable now on the existing checkpoint (no GPU):** load
`embedding_preprocessor` from `best_metric.ckpt`, push the 6,607 stored embeddings through it,
and compare cosine similarity in the 3,328-d input space to cosine similarity in the 90-d
output space over random gene pairs. If the map is smooth the two correlate; if it is acting
as a lookup table they decorrelate. I could not run this in-session because
`cell_graph["gene"].x` requires building the fig3_core dataset; the checkpoint half is ready
(spectra already extracted: `W1` eff rank 36.2, `W2` eff rank 54.3).

### A4 -- Give the observed values an undiluted route, and widen the bottleneck

**Change (bounded code, ~20 lines):** in `CellGraphTransformer.forward` around `:3024-3041`,
add a strain-level summary of the observed labels that does NOT go through the 6,607-key
softmax: project the `[B, N]` observed-value vector with a single `Linear(N, r)` (r = 64) and
add the result to `c_b` / to every token, alongside the existing per-gene injection. Plus a
config-only companion arm `model.post_perturbation_mixing.num_latents=128`.

**Hypothesis:** `PerceiverMixing.write_attn` pools 10 revealed genes among 6,607 keys, so the
conditioning signal is attenuated ~660x before it can reach the latents. Measured symptom: the
model scores 0.215 / 0.250 / 0.326 at 10 / 100 / 1000 revealed genes against a plain ridge
oracle's 0.408 / 0.676 / 0.793.

**Diagnostic, cheap on an existing checkpoint:** run one validation batch at k=1 (10 revealed)
with and without the `ObservedLabelEncoder` contribution zeroed and measure the change in the
32 latents and in the predictions. If the latents move by less than a few percent, the
dilution mechanism is confirmed. This needs the dataset, so it is a GPU-free but
dataset-dependent probe.

### A5 -- Fix the two places the strain signal is corrupted or the run dies

**Change (config only), three keys:**
```
model.perturbation_head.dropout=0.0
regression_task.lr_scheduler.type=CosineAnnealingWarmupRestarts  (warmup_steps ~ 2000)
model.perturbation_head.residual=rezero
```

**Hypotheses, one per key:** (a) at |S| = 1 the attention weight is exactly 1.0, so
`dropout=0.1` deletes the perturbation context entirely for 10% of gene tokens every step and
rescales the rest by 1.111, a train/eval mismatch on the only strain-conditional signal;
(b) no warmup with post-LN produced 1 dead run in 24 in v13 and 3 collapses in 16 among the
v12 arms that add zero-gated branches; (c) ReZero keeps the residual stream unnormalized so
`||h_i + c_b||`, where `<h_i, c_b>` lives, is not discarded.

**Diagnostic, already available:** the collapse rate itself (1/24 in v13, 3/16 in the
branch-adding v12 arms, 0/16 in the four clean arms). For (a), a 5-minute CPU probe of
`EquivariantPerturbationTransform` in `.train()` vs `.eval()` at |S| = 1 measuring the fraction
of gene rows whose `attended` is exactly zero; this reuses
`scripts/perturbation_selector_degeneracy.py` unchanged except for the dropout setting.

---

## 10. Things I would NOT do

- **Do not add graph-based pair routing** (`P_graph`, GEARS-style propagation over the nine
  graphs). `results/graph_prior_probe.json` measured all nine at chance. The one exception
  worth a config change is stopping the symmetrization of `tflink` and
  `regulatory_interaction` in `_build_head_mask` `:2617-2618`, where the directed form reads
  0.5508 / 0.5239 against 0.5057 / 0.5055 symmetrized.
- **Do not widen the output rank yet.** The model is below a rank-1 oracle; rank is not the
  binding constraint at 0.22. It becomes binding above ~0.53 (the rank-4 train-basis ceiling).
- **Do not re-run the null sink.** It supplies a rank-9 pair term; the Perceiver already
  supplies a larger one and the prediction matrix is rank 4.8, so nothing is rank-starved.
- **The metabolic module will not help this.** The failure is that the model cannot mark which
  gene was deleted and cannot assign per-strain amplitudes to more than ~5 response programs.
  A flux layer changes neither. (Consistent with memory `flux-layer-collapses-every-readout`.)

---

## Summary (600 words)

The v13 expression model is 6.67M parameters of which **96.4% are strain-invariant**: 5.85M in
a gene-embedding preprocessor, 590K in a six-layer encoder that runs at **batch 1 on the
wildtype graph before the perturbation exists**. The entire genotype-conditional path is one
98K-parameter cross-attention block (1.5%), and the readout is a 9,919-parameter MLP shared
across all 6,127 output genes (0.15%). At |S| = 1, which is 95.4% of the build, that
cross-attention softmaxes over a single key, so the deletion becomes one 90-d vector added
identically to every gene token. Genes interact densely in the encoder (five of six layers are
unmasked; layer 1 is masked to graph neighbors on all nine heads), but none of that is
strain-dependent. After the deletion, the only gene-gene channel is a 32-latent Perceiver.

Every pair-term mechanism ever written into the file is **off** in the round that produced the
headline numbers: propagation, null sink, Hadamard, response basis, bilinear, FiLM, the GEARS
per-gene row, the State context row, GEARS pooled mixing.

Diagnostics I ran on existing dumps and checkpoints:

- The val prediction matrix has **effective rank 4.80**; a rank-5 truncation reproduces the
  full score (0.2280 vs 0.2227). A **rank-1 oracle scores 0.317**. The model is worse than a
  rank-1 oracle, so output rank is not what is costing points today. Proteome predictions are
  effective rank 1.35-2.11 against a target rank of 36-47.
- The deleted gene's own log2 ratio averages **-2.42 (8.26 sd below its column mean)** and is
  the largest |change| in the true profile for the median strain. The model predicts **+0.036**
  and ranks it at the **49.5th percentile** of its own predicted changes. Substituting a
  constant -1.0 at that one cell per strain, with no retraining, moves **val 0.2227 -> 0.2366
  and test 0.1095 -> 0.1241**. The hop-0 self-indicator that supplies this exists in the code
  and is disabled; its one prior test ran with a closed gate at n=1.
- The masked objective is paid for and unlearned. Only 25% of steps are k=0, each step pays an
  extra no-grad forward, and at the end `traineval@k0 = 0.7729` vs `@k3 = 0.7740`: revealing
  1,000 true labels changes the train fit by +0.001. On val the imputation peaks at **0.482 at
  epoch 1201** and decays to **0.317**. A ridge on 10 revealed genes scores 0.408, beating the
  model on 1,000 (0.326).
- **val@k0 is flat from epoch 2000** (0.201 at 2001, 0.198 at 4001, max 0.2138 at 3641), so
  epochs 2000-6000 buy nothing on the headline metric.
- 1 of 24 v13 runs never learned (val and traineval both ~0.00 for 4,000 epochs); 3 of 16 v12
  runs collapsed, all in arms that add zero-gated branches to the post-LN block.
- Correction: the v12 per-gene head measured **-0.0084** at the matched 1,399-epoch budget, not
  +0.041. The +0.03 figure is the 500-epoch contrast (per-gene +0.0319, State +0.0356, both
  4/4 positive), which washes out by 1,400. Per-gene output rows are a **speed** lever.
- Correction: the graph channel's justification (L-step reachability from the deleted gene)
  describes an architecture that is not implemented, and `graph_prior_probe.json` already
  measured all nine graphs at chance.

Five arms proposed: (1) hop-0 self-indicator with the gate forced on, config-only, with a
measured +0.014 lower bound; (2) drop or reweight the mask schedule, ~6x more on-task gradient
per hour; (3) collapse the preprocessor to `num_layers=1` (5.85M -> 300K), config-only;
(4) an undiluted route for observed values plus 128 latents, ~20 lines; (5)
`perturbation_head.dropout=0`, warmup, and ReZero, config-only.

---

## Files and paths

```
/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/agent-03-architecture.md
/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/rank.py
/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/rank2.py
/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/rank3.py
/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/align2.py
/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/selfconst.py
/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/selforacle.py
/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/lossweight.py
/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/ckpt.py
/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/torchcell/models/equivariant_cell_graph_transformer.py
/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py
/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_v13_split.yaml
/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/graph_prior_probe.json
/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/masked_conditioning_oracle.json
/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/head_round_readout.json
/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/lowrank_output_ceiling.json
/scratch/projects/torchcell-scratch/val-predictions/compute-0-2-2397312_3fcfb279cc58c21f926585737901556cf40cded9e6c3fc039c1b34abbc9b7248.json
/scratch/projects/torchcell-scratch/models/checkpoints/compute-0-2-2397312_3fcfb279cc58c21f926585737901556cf40cded9e6c3fc039c1b34abbc9b7248/best_metric.ckpt
```

W&B runs cited, one URL per line:

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/wq8y8nd5

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/825on260

https://wandb.ai/zhao-group/torchcell_019_prot_v14/runs/uc0pm2pv

https://wandb.ai/zhao-group/torchcell_019_expr_v12/runs/kjs3u9rw
