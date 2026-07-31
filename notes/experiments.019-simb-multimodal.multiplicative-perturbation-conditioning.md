---
id: n1kjowg1xg6561hudy5w2hh
title: Multiplicative Perturbation Conditioning
desc: ''
updated: 1785388392865
created: 1785388392865
---

## 2026.07.30 - Multiplicative Perturbation Conditioning (M_op, M_dec, FiLM)

Design record for the MULTIPLICATIVE / HADAMARD class of perturbation conditioning in
019. Two designs: `M_dec` (multiplicative at the readout -- already implemented, one
defect) and `M_op` (multiplicative at the perturbation operator -- not implemented). No
training was run for this note; every number below is attributed.

**Provenance caveat, stated up front.** Three classes of number appear here:
(a) values read from committed result artifacts under
`/home/michaelvolk/Documents/projects/torchcell.worktrees/019-decoder-pair-routing/experiments/019-simb-multimodal/results/`;
(b) values I re-derived from those artifacts in this session (arithmetic shown);
(c) values measured by an ad-hoc init-time probe of the real module that is **not yet a
committed script**, which violates the repo's artifact rule -- landing that probe under
`experiments/019-simb-multimodal/scripts/` is listed in Next Steps.

### 1. The problem: the perturbation enters additively, exactly once

Today the strain enters the model at ONE site, by ADDITION:

```
H_pert_i = g(h_i + c_b)
```

- `EquivariantPerturbationTransform.forward` computes `attended` (the context `c_b`) by
  cross-attention whose keys/values are ONLY the perturbed tokens
  (`equivariant_cell_graph_transformer.py:644`, `:662-667`), then fuses it into every gene
  by addition inside `_apply_residual` -- post-LN `norm1(x + dropout(attended))` at
  `:591-593`, ReZero `x + beta_attn * dropout(attended)` at `:584`.
- The module's own docstring states the consequence: `h_i` and `c_b` meet exactly once, by
  addition, so any downstream head sees only a function of their SUM (`:626-630`).
- Everything upstream is strain-INDEPENDENT: the encoder runs at batch 1 on the wildtype
  graph (`:2622` builds `X` as `[1, N+1, d]`), and `h_CLS` / `H_genes` are sliced from that
  single row (`:2688-2690`).

So `c_b` is one `d = 90` vector shared by all `N = 6607` gene tokens. **No term in the
baseline forward pass depends on the pair (perturbation `p`, measured gene `i`).**

#### 1.1 Measured: at |S| = 1 the selector is degenerate

Source: `experiments/019-simb-multimodal/scripts/perturbation_selector_degeneracy.py` ->
`results/perturbation_selector_degeneracy.json` (`hidden_dim` 90, `num_heads` 9,
`num_genes` 6607, seed 0).

| quantity | value | key in JSON |
|---|---|---|
| distinct attention weights at \|S\|=1 | `{1.0}` | `s1_weights_unique` |
| across-query spread of the attention weight | `3.31e-09` | `s1_query_spread` |
| output change when `W_Q`, `W_K` are re-drawn at std=10 | **exactly `0.0`** | `s1_wq_wk_ablation_max_change` |
| dead attention params | **16,200** of **32,760** | `s1_dead_params` / `s1_total_attn_params` |
| \|S\|=2 weight range (for contrast) | `0.0218 - 0.9782` | `s2_weight_min/max` |

With `|S| = 1` the softmax runs over a ONE-element key set, so `alpha = 1.0` for every
query and the attended context is query-independent; `c` collapses to an affine map of the
deleted gene, `c = W_O(W_V h_p + b_V) + b_O` (module comment `:441-442`, verified there to
`2.4e-7`). **95.4% of the expression build is `|S| = 1`** (`conf/cgt_expr_008.yaml:161`,
module `:444`), so those 16,200 parameters are dead weight on 95.4% of the data.

*Bookkeeping discrepancy worth fixing:* the module comment at `:443` and the arm script at
`gh_expr_008_arm.sh:161` both say **16,380**, while the committed JSON says **16,200**. Both
are arithmetically correct under different accounting -- `2 * 90 * 90 = 16,200` is the
`W_Q`/`W_K` WEIGHTS only; adding `b_Q` and `b_K` (`2 * 90 = 180`) gives 16,380. The JSON is
the measured artifact, so quote 16,200 and say "weights"; the two prose sites should be
reworded rather than left inconsistent.

#### 1.2 Measured: the null sink was the rank-9 fix, and it came back null -- UNDERPOWERED, not established-null

`null_sink` appends one zero-valued K/V column carrying a learned logit offset, making
`alpha_i = sigmoid(q_i . k_p / sqrt(d_k) - null_bias)`, which finally depends on `i`
(design + construction: `:436-501`, `:650-661`). Wave 4b ran the magnitude-matched version
(`N_sink_mm`, `gh_expr_008_arm.sh:278-289`) against a co-launched in-wave reference
`R_ref`, 4 seeds (0, 1, 2, 42).

Re-derived by me this session from `results/wave_convergence_stage-wave4b.csv` (`roll_max`
column) and cross-checked against `results/decoder_arms_torchcell_019_expr_v8.csv`
(`smoothed`, wave4b block) -- the two agree:

| statistic | value |
|---|---|
| per-seed paired delta (seeds 0/1/2/42) | `-0.0011`, `-0.0081`, `+0.0064`, `+0.0125` |
| paired mean | **+0.0024** |
| paired sd | **0.0090** |
| t(3) 95% CI | **+/- 0.0143** (contains 0) |
| n | **4** |

W&B run ids: `N_sink_mm` = `v1grwe46` / `kyvkrjcb` / `047tg3h4` / `y42h0pwu`; `R_ref` =
`q3s1u8j5` / `7zrt6udr` / `wwh2eio6` / `bdloc0mw`. (These runs were launched by hand via
`gh_expr_008_arm.sh`, not sbatch; the launch was labelled "job 1413" in session notes but no
slurm record for it exists in this worktree, so the run ids above are the citable handle.)

**This is UNDERPOWERED, not established-null.** Two independent reasons:

- **MDE at 80% power is `+0.0191` = 7.9x the point estimate** at `n = 4` under the
  wave-1..4 seed scheme (`conf/cgt_expr_011.yaml:19`). Recomputing from the sd above with
  `(t_{0.975,3} + t_{0.80,3}) * se = (3.182 + 1.638) * 0.0045` gives `+0.0216` (9.0x) --
  same conclusion, slightly more conservative convention.
- **Exact paired tests floor at `p = 0.125` at `n = 4`.** A two-sided sign test's smallest
  attainable p-value is `2 / 2^4 = 0.125`, so no arrangement of 4 paired seeds can reach
  `p < 0.05` regardless of effect size.

Root cause of the low power, and the reason wave 5 changed the design: under the old scheme
`seed` drove BOTH weight init and the train/val partition, giving between-seed sd `0.0444`
against a within-seed across-arm sd of `0.0058` -- the nuisance axis was `7.7x` the signal
axis (`conf/cgt_expr_011.yaml:10-15`). `_011` pins `data_module.split_seed: 0`
(`cgt_expr_011.yaml:53-56`) so `seed` varies init only.

#### 1.3 The central argument: CAPACITY ACCOUNTING

At `|S| = 1` with 9 heads, the entire pair-dependent signal that can reach gene `i` is
**9 bounded scalars** (one attention weight per head) set against a 90-d gene-independent
context. Per head the induced per-gene variation is rank-1; across heads it is rank
`num_heads` (module `:455-457`).

Measured by an init-time probe of the real `EquivariantPerturbationTransform`
(`hidden_dim=90, num_heads=9, dropout=0, null_sink=True, bias_init=0.0,
magnitude_match=True`, `|S|=1`, `H_genes ~ N(0,1)`; probe class (c) -- not yet committed).
Ran `N in {2048, 6607} x seed in {0,1,2}`:

| quantity | measured | draw-dependence |
|---|---|---|
| numerical rank of the gene-dependent context variation `c - mean_i c` | **exactly 9** in all 6 draws | INVARIANT (= `num_heads`) |
| `sv[9] / sv[8]` | `1.25e-06` to `1.96e-06` | draw-dependent; previously reported `1.16e-6` sits just under this range |
| pair-dependent share of context energy | `4.75%` to `5.90%` | draw-dependent; previously reported `6.43%` is just above |
| same, with the sink OFF | `3.7e-12 %`, rank 3 (float noise) | -- |
| `b_K` gradient rms | `8.3e-17` (abs-max `2.8e-16`) over its **90** params | INVARIANT: `b_K` enters BOTH logits identically and cancels in the softmax, so it is exactly and permanently dead |
| grad rms, value/out path (`W_V`, `W_O`) vs query/key path (`W_Q`, `W_K`) | `1.67e-08` vs `6.33e-10` = **26.4x** on a plain `mean(out^2)` probe loss | loss-dependent; the previously reported `281.8x` used a different probe loss. The ORDERING (value path dominates the revived q/k path by 1-2 orders of magnitude) reproduces; the multiplier does not transfer between losses and should not be quoted without its loss. |

**The rank ladder, which is the whole motivation:**

| channel | pair-(p, i) rank at \|S\|=1 | status |
|---|---|---|
| additive context `g(h_i + c_b)` (baseline) | **0** | measured degenerate (1.1) |
| null sink (`alpha_i` per head) | **9** | measured null at n=4, underpowered (1.2) |
| multiplicative / FiLM `gamma(c_b)` over `d` channels | **90** | UNTESTED |

10x the capacity of the mechanism that returned an underpowered null. That ratio -- not any
prior -- is the argument for running the multiplicative arms.

### 2. `M_dec` -- multiplicative at the READOUT (implemented; one defect)

**Already built.** `PerGeneHead.__init__` takes `film_dim` and, when `film_dim > 0`, builds
a conditioner `Linear(film_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 2 * width)` whose
FINAL layer has weight AND bias zero-initialized (`:1560-1570`, with `width = hidden_dim *
in_mult + free_gene_dim + extra_dim` at `:1562` so it matches whatever the concat/bilinear/
free-gene arms widened the input to). Application in `forward`:

```python
gamma, beta = self.film(film_cond).chunk(2, dim=-1)                       # :1603
H_genes_pert = H_genes_pert * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # :1604
```

**Identity at init, verified by reading:** the final `Linear` is zeroed at `:1569-1570`, so
`gamma = beta = 0` at step 0 and the `1 + gamma` form makes `:1604` the exact identity
`H * 1 + 0`. That is what makes the arm a CLEAN ABLATION -- an arm that is not an identity
at init confounds "the mechanism helps" with "the block perturbed the init", which is
exactly the confound that made the unmatched null sink uninterpretable (module `:470-490`:
`||c_sink||/||c_ref|| = 0.5139`, and a bare scalar `0.5` on the reference reproduced 97.8%
of the arm's downstream deviation). FiLM also engages only when BOTH `film_dim > 0` and
`film_cond is not None` (`:1599`), so `film_cond=None` reduces the head to its
unconditioned form -- a free sham path.

Wiring: `film_dim = 2 * hidden_channels` iff `film_on_pert_set` (`:2131-2133`),
`self.per_gene_film` at `:2143`, config key declared at `conf/cgt_expr_008.yaml:93`
(`film_on_pert_set: false`), and the arm already exists as **`F2_film`** ->
`multitask.film_on_pert_set=true` (`gh_expr_008_arm.sh:90`).

#### 2.1 DEFECT to fix before `M_dec` is run

`pert_cond` is built as `cat([z_S, h_CLS.expand(bsz, -1)])` at `:2797` and passed as
`film_cond` at `:2805-2806`. `h_CLS` is a single `[d]` vector sliced from the batch-1
wildtype encoder pass (`:2689`), so the `expand` at `:2797` **broadcasts one identical
vector across every strain in the batch** -- its across-strain variance is identically zero
by construction (verified by reading; `expand` creates a stride-0 view, no per-strain
content exists to vary). Measured confirmation reported previously: across-strain sd `0.0`
for the `h_CLS` half vs `0.973` for the `z_S` half (class (c) -- ad-hoc, uncommitted).

Consequence: **half of the conditioner's 180-d input carries no strain information** and can
only contribute a constant offset to `gamma`/`beta`. The mechanism is not broken, but its
input is half-wasted, and a null from this configuration would be uninterpretable in the
same way A4's was.

**Fix:** pass `z_S` only, `film_dim = hidden_channels` (90, not 180). `z_S` is pooled over
`H_genes_pert[b, sel, :]` (`:2788-2796`), i.e. over the POST-perturbation representation of
the perturbed genes, so it is genuinely strain-dependent. Note the same
`pert_cond` tensor also feeds `pert_set_context` (`:2798-2802`) and `response_basis`
(`:2808-2815`), so the change must either be local to the FiLM call site or be a deliberate
change to all three (they are separate arms and should not be silently coupled).

### 3. `M_op` -- multiplicative at the OPERATOR (not implemented)

**Design.** Inside `EquivariantPerturbationTransform`, replace or augment the additive
injection with a Hadamard ASSERTION:

```
H_pert_i = g(h_i * (1 + gamma(c_b)))          # optionally  + beta(c_b)
```

i.e. the perturbation multiplicatively rescales every gene's 90 channels, rather than being
retrieved by query-key similarity and added. This is the closer reading of "Hadamard
assertion instead of cross-attention": the strain is **asserted into the channels** of every
gene token, so the pair term is `h_i` -> `h_i * gamma(c_b)`, whose dependence on `i` is the
full elementwise product rather than 9 scalars.

Requirements, all of them consequences of things already measured above:

1. **Identity at init.** Zero-init the final layer of the `gamma` generator and use the
   `1 + gamma` form -- exactly the pattern `PerGeneHead` already uses at `:1569-1570` /
   `:1604`. Without this the arm inherits the null sink's attenuation confound.
2. **Insertion point is a choice, and it changes the rank story.** Applying `gamma` to
   `h_i` BEFORE `_apply_residual` keeps the existing additive path intact and adds a
   rank-`d` multiplicative one (augment); replacing the additive injection entirely is a
   different arm and should be tagged separately.
3. **`gamma` must be generated from a strain-dependent input.** `c_b` at `|S| = 1` is an
   affine map of `h_p` (see 1.1), which IS strain-dependent -- so `gamma(c_b)` is legitimate
   here. Do not substitute `h_CLS` (see 2.1).
4. **Post-LN erases magnitude.** With `residual: postln` the block closes with a LayerNorm
   (`:591-593`), and LayerNorm is invariant to positive rescaling of its input (module
   `:402-405`), so a UNIFORM `gamma` would be partly annihilated. Only the
   per-channel-varying part of `gamma` survives. Either run `M_op` with `residual: rezero`
   (`:581-590`, exact identity at init) or state plainly that post-LN caps what the arm can
   express.
5. **Engagement logging.** `gamma` abs-max plus a per-strain sd of `gamma`, per epoch --
   see Next Steps.

### 4. Why not just zero out the deleted gene's embedding

This question keeps recurring; recording the answer with its evidence.

**Zeroing row `p` of `H_genes_pert` (i.e. at the decoder input) changes exactly ONE
prediction of 6607.** Verified by reading the readout path: nothing between the perturbation
transform and the per-gene prediction mixes genes unless explicitly enabled. `PerGeneHead`
is a shared MLP applied to the last dim of `[batch, N, width]` (`:1571-1576`, `:1605`), the
free-gene embedding is a row-wise concat (`:1595-1598`), and FiLM broadcasts per-strain over
the gene axis (`:1604`). The module docstring states the same structural fact: after the
perturbation there is NO gene-gene interaction; gene `i`'s state depends on `(h_i, {h_p})`
and never on `h_j` (`:1097-1104`, `:2731-2733`). And the encoder ran BEFORE the perturbation
(`:2622`, `:2688-2690`), so it cannot propagate the edit either.

Each val strain scores 6127 measured genes (`conf/cgt_expr_008.yaml:32`,
`train_cgt_multitask.py:261`), of which the deleted gene is at most 1, so **~99.98%
(6126/6127) of scored predictions do not move.**

**Measured, and this is the direct test:** the `A2_self` arm IS this mechanism -- a hop-0
self-indicator, "is gene `i` itself the deleted gene", one graph-independent feature
(`PerturbationGraphPropagation`, `:753-762`, `:811-812`; arm at `gh_expr_008_arm.sh:40`,
`hops=0`, which yields `num_features = 1` at `:762`). Paired delta vs the in-wave
`A0_baseline` at seed 0, wave1, from `results/decoder_arms_torchcell_019_expr_v8.csv`
(`smoothed`, the scoring column `score_decoder_arms.py:267` uses):

| metric | A0_baseline | A2_self | delta |
|---|---|---|---|
| `smoothed` (the scored statistic) | `0.152654` | `0.151651` | **-0.0010** |
| `raw_peak` | `0.169634` | `0.172939` | `+0.0033` |
| `last` | `0.085103` | `0.148050` | `+0.0629` |

n = 1 (seed 0 only), so **no significance claim**; all three are far inside the 0.0444
between-seed sd of the old scheme. A previously circulated figure of `+0.00005` for this arm
does NOT reproduce from the committed CSV under any of its scoring columns -- use `-0.0010`
(`smoothed`) and say n=1.

**CORRECTION to a claim made while specifying this note: A2_self's gate ENGAGEMENT is NOT
established.** `A2_self` inherits the default `gate_mode: rezero`
(`conf/cgt_expr_008.yaml:108-115`, model default at `:734`, wired at `:2076`), so the
propagation gate starts at ZERO -- closed -- and the arm is an exact identity at init
(`:776`, conf `:39-40`). Gate scalars were only added to W&B later
(`train_cgt_multitask.py:1342-1375`, matching by suffix `("null_bias", "gate", "beta_attn",
"beta_ffn", "ablate_mask")` at `:1355`), and the wave1 rows of the v8 CSV carry no gate
columns -- only `wave_convergence_stage-wave4b.csv` has `gate_first` / `gate_end` /
`gate_absmax`. So for THIS run engagement is unmeasured. The nearest evidence is the conf
note that a propagation gate reached `0.125` while the Perceiver's finished at `1.0e-5`
under the same mechanism (`conf/cgt_expr_008.yaml:112-114`) -- suggestive that propagation
gates do open, not proof about `A2_self`. Read the arm as **n=1 and engagement-unverified**,
which weakens it as evidence and is a second reason the self-indicator question deserves a
re-run with gate logging on.

**The one condition under which zeroing WOULD matter:** a pathway that routes information
from row `p` to row `i` AFTER the perturbation. That pathway exists as
`post_perturbation_mixing` / `PerceiverMixing` (`:1094-1120`, called at `:2734-2735`) and as
`CrossGeneMixing` (`:2727-2729`). Measured: `E0_perceiver_on` **-0.0302** and
`GEARS_crossgene` **-0.0243**, both **n = 1, seed 0** -- no significance, and both taken
before cross-seed comparison was known to be invalid (`gh_expr_008_arm.sh:220-223`;
deltas reproduce from the v8 CSV, wave3 vs `A1_ref`). H2 in wave 5 re-tests exactly this
with 9 paired replicates.

**Also worth recording, because it is the subtler half of the question:** zeroing `h_p`
BEFORE the perturbation transform is NOT a one-row edit -- `h_p` IS the key/value of the
cross-attention (`:644`), so it changes `c_b`, hence every gene. But it changes `c_b` only
through the SHARED context, which at `|S| = 1` is gene-independent (1.1). So that variant
cannot create a pair-(p, i) term either; it only moves the rank-0 channel. Neither zeroing
site buys pair-dependence.

### 5. A rank ladder as the discriminating experiment

If pair-term CAPACITY is really the binding constraint, the ladder in 1.3 makes a falsifiable
prediction: **a `gamma` restricted to rank 9 should reproduce the null sink's null, and rank
90 should not.** If rank-9 and rank-90 `gamma` score the same, capacity is NOT the binding
constraint and the multiplicative family should be dropped rather than widened.

HYPOTHESIS (untested): the above prediction. Nothing in 019 has measured a rank-restricted
multiplicative conditioner.

Proposed contrast -- `gamma` rank in `{9, 30, 90}`, implemented as a low-rank factorization
of the `gamma` generator's final layer (`U V^T` with inner dim `r`, `U` zero-init to keep
identity-at-init), run as a WITHIN-POOL comparison against a co-launched reference:

| arm | `gamma` rank | reads as |
|---|---|---|
| `M_r9` | 9 | matched to the null sink's capacity -- should reproduce its null |
| `M_r30` | 30 | interpolation; is the response monotone in rank? |
| `M_r90` | 90 | full elementwise -- the 10x arm |
| reference | -- | co-launched, same pool |

**Hardware homogeneity is a hard requirement, not a nicety.** Arms compared against a
paired reference must run on the SAME GPU type and the SAME packing: `_007` pooled A100
(mmli) with RTX 6000 Ada (GilaHyper) runs and their means differ, `0.0242` vs `0.0182`, so
every wave-5 pool gets its own co-launched `W_ref` rather than sharing one
(`gh_expr_008_arm.sh:193-199`). Same rule applies to the rank ladder: one reference per
pool, or the denominator is invalid.

### 6. Status / next steps

**Status: SPECCED, then DROPPED from wave 5.** `M_op` and `M_dec` were specified and then
cut from the wave-5 round in favour of two other hypotheses, which is what
`gh_expr_008_arm.sh:188-265` actually launches:

- **H1 -- is training saturated?** `H1_ref` / `H1_nodrop` (`:210-213`), read via the new
  eval-mode train metric `traineval/expression/pearson_per_feature`
  (`trainer.train_eval_every: 1`, `conf/cgt_expr_011.yaml:27-37`, `:58-59`). The existing
  `train/...` series is accumulated with dropout active over weights still updating
  in-epoch, i.e. biased low, which is why "is it saturated?" was never answered.
- **H2 -- does post-perturbation cross-gene mixing help?** `X_mix` / `X_mix64` /
  `X_mix_rezero` (`:230-243`), largest pool, 9 paired replicates.
- Plus config-only probes: `G_l3` / `G_l13` mask depth (`:254-257`), `P_lr1e4` / `P_lr3e5`
  (`:262-265`).

**The rank argument in 1.3 is unaffected by that decision.** It is an init-time property of
the module, not a claim about wave-5 outcomes. Neither H1 nor H2 measures it: H1 is a
capacity question about the optimizer/regularizer, H2 opens the gene-gene channel (a
different missing structure -- cascades, not pair-dependence).

**Exact remaining work:**

1. **Fix `film_cond` to `z_S`-only** -- `:2797` currently concatenates the stride-0
   `h_CLS` expand; set `film_dim = hidden_channels` at `:2131-2133`. Do not silently change
   `pert_set_context` / `response_basis`, which share the same `pert_cond` tensor.
2. **Implement `M_op`** in `EquivariantPerturbationTransform` per section 3 -- zero-init
   final layer, `1 + gamma` form, insertion point declared as its own arm, and decide
   post-LN vs ReZero explicitly.
3. **Add ENGAGEMENT LOGGING to both**: `film/gamma_absmax` and a **per-strain sd of
   `gamma`** (and the same for `M_op`), logged every train epoch to W&B. This is the
   lesson from A4: its paired delta was `~0` and looked like a clean negative, but reading
   the four `last.ckpt` at epoch ~187 showed `null_bias` had moved only `-4.0 -> -3.92/-3.95`
   -- the mechanism was **never in the forward pass**
   (`gh_expr_008_arm.sh:266-273`). A per-strain sd of `gamma` distinguishes "no strain
   information reached the conditioner" (sd ~ 0 -> the 2.1 defect, or a dead arm) from
   "the mechanism engaged and did not help" (sd > 0, delta ~ 0). Without it a dead arm is
   reported as a null.
4. **Land the init-time capacity probe as a committed script** under
   `experiments/019-simb-multimodal/scripts/` (writing its JSON to `results/`), so the
   rank-9 / energy-share / `b_K`-dead numbers in 1.3 stop being class-(c) values. It should
   emit: `sv` spectrum of `c - mean_i c`, numerical rank, pair-energy share, `b_K` grad, and
   the per-path grad ratio WITH its loss named.
5. **Reconcile the 16,380 vs 16,200 dead-parameter count** in the two prose sites
   (`equivariant_cell_graph_transformer.py:443`, `gh_expr_008_arm.sh:161`) against the
   committed JSON (see 1.1).

### Related

- [[experiments.019-simb-multimodal.fig3-expression-experiments]]
- [[experiments.019-simb-multimodal.decoder-distributional-plan]]
- [[experiments.019-simb-multimodal.experimental-plans]]
- [[torchcell.models.equivariant_cell_graph_transformer]]
